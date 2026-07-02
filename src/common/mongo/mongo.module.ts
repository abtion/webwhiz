import { Injectable, Module, OnApplicationShutdown } from '@nestjs/common';
import { MongoClient, Db } from 'mongodb';
import { AppConfigService } from '../config/appConfig.service';

export const MONGODB = 'MONGODB';
const MONGO_CLIENT = 'MONGO_CLIENT';

@Injectable()
class MongoClientService implements OnApplicationShutdown {
  constructor(private readonly client: MongoClient) {}

  async onApplicationShutdown() {
    await this.client.close();
  }
}

@Module({
  providers: [
    {
      provide: MONGO_CLIENT,
      inject: [AppConfigService],
      useFactory: async (appConfig: AppConfigService): Promise<MongoClient> => {
        const mongoUri = appConfig.get('mongoUri');
        const client = new MongoClient(mongoUri);
        await client.connect();
        return client;
      },
    },
    {
      provide: MongoClientService,
      inject: [MONGO_CLIENT],
      useFactory: (client: MongoClient) => new MongoClientService(client),
    },
    {
      provide: MONGODB,
      inject: [MONGO_CLIENT, AppConfigService],
      useFactory: async (
        client: MongoClient,
        appConfig: AppConfigService,
      ): Promise<Db> => {
        const dbName = appConfig.get('mongoDbName');

        try {
          const db = client.db(dbName);

          // Create indexes
          await db
            .collection('users')
            .createIndex({ email: 1 }, { unique: true });

          await db
            .collection('kbDataStore')
            .createIndex({ knowledgebaseId: 1, type: 1 });

          await db.collection('chunks').createIndex({ knowledgebaseId: 1 });
          await db.collection('chunks').createIndex({ dataStoreId: 1 });

          await db
            .collection('kbEmbeddings')
            .createIndex({ knowledgebaseId: 1 });

          await db.collection('knowledgebase').createIndex({ owner: 1 });
          await db
            .collection('knowledgebase')
            .createIndex({ customDomain: 1 }, { sparse: true, unique: true });

          await db
            .collection('offlineMessages')
            .createIndex({ knowledgebaseId: 1 });

          await db
            .collection('chatSessions')
            .createIndex({ knowledgebaseId: 1 });
          await db.collection('chatSessions').createIndex({ slackThreadId: 1 });

          await db.collection('task').createIndex({ name: 1 });

          await db
            .collection('slackTokens')
            .createIndex({ teamId: 1 }, { unique: true });

          return db;
        } catch (e) {
          throw e;
        }
      },
    },
  ],
  exports: [MONGODB],
})
export class MongoModule {}
