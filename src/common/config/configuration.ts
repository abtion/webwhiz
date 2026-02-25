import { EmbeddingModel } from '../../knowledgebase/knowledgebase.schema';
import { Subscription } from '../../user/user.schema';

interface AppConfig {
  nodeEnv: 'development' | 'production' | 'test';
  host: string;
  port: number;
  secretKey: string;
  encryptionKey: string;
  mongoUri: string;
  mongoDbName: string;
  redisHost?: string;
  redisPort?: number;
  redisUrl?: string;
  defaultEmbeddingsModel?: string
  defaultChatModel?: string
  aiProvider: 'openai' | 'openai-azure'
  openaiKey: string;
  openaiKey2: string;
  openaiAzureEndpoint: string;
  openaiAzureKey: string;
  openaiAzureVersion: string;
  googleClientId: string;
  lemonSqueezyApiKey: string;
  lemonSqueezySignSecret: string;
  sendGridApiKey: string;
  docStorageLocation: string;
  textractServiceUrl: string;
  defaultSubscription: Subscription;
  senderEmail: string;
  senderName: string;
  clientUrl: string;
}

const config = (): AppConfig => ({
  nodeEnv: (process.env.NODE_ENV as any) || 'development',
  host: process.env.HOST || '127.0.0.1',
  port: parseInt(process.env.PORT as string, 10) || 8000,
  secretKey: process.env.SECRET_KEY || 'secret',
  encryptionKey:
    process.env.ENC_KEY ||
    '4b23f673b133b01777259525a4a1c2097fe309048db6593462312d5c35679ced',
  mongoUri: process.env.MONGO_URI || 'mongodb://127.0.0.1:27017',
  mongoDbName: process.env.MONGO_DBNAME || 'sitemine',
  redisHost: process.env.REDIS_HOST || '127.0.0.1',
  redisPort: parseInt(process.env.REDIS_PORT as string, 10) || 6379,
  redisUrl: process.env.REDIS_URL,
  defaultEmbeddingsModel: process.env.DEFAULT_EMBEDDINGS_MODEL || EmbeddingModel.OPENAI_EMBEDDING_3,
  defaultChatModel: process.env.DEFAULT_CHAT_MODEL || 'gpt-3.5-turbo',
  aiProvider: (process.env.AI_PROVIDER || 'openai') as AppConfig['aiProvider'],
  openaiKey: process.env.OPENAI_KEY || '',
  openaiKey2: process.env.OPENAI_KEY_2 || '',
  openaiAzureEndpoint: process.env.OPENAI_AZURE_ENDPOINT || '',
  openaiAzureKey: process.env.OPENAI_AZURE_KEY || '',
  openaiAzureVersion: process.env.OPENAI_AZURE_VERSION || '',
  googleClientId: process.env.GOOGLE_CLIENT_ID || '',
  lemonSqueezyApiKey: process.env.LEMON_SQUEEZY_API_KEY || '',
  lemonSqueezySignSecret: process.env.LEMON_SQUEEZY_SIGN_SECRET || '',
  sendGridApiKey: process.env.SENDGRID_API_KEY || '',
  docStorageLocation: process.env.DOC_STORAGE_LOCATION || '../storage',
  textractServiceUrl:
    process.env.TEXTRACT_URL || 'http://localhost:8080/textract',
  defaultSubscription:
    Subscription[process.env.DEFAULT_SUBSCRIPTION] || Subscription.FREE,
  senderEmail: process.env.EMAIL_SENDER_EMAIL || 'hi@webwhiz.ai',
  senderName: process.env.EMAIL_SENDER_NAME || 'WebWhiz.ai',
  clientUrl: process.env.CLIENT_URL || 'https://app.webwhiz.ai',
});

export { AppConfig };
export default config;
