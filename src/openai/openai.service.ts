import { Injectable, Logger } from '@nestjs/common';
import OpenAI, { AzureOpenAI } from 'openai';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import { Subject } from 'rxjs';
import { AppConfigService } from '../common/config/appConfig.service';
import * as tiktoken from '@dqbd/tiktoken';
import { createHash } from 'node:crypto';
import { EmbeddingModel } from '../knowledgebase/knowledgebase.schema';
import { APIError } from 'openai/error';

const TOKENIZERS = {
  chatgtp: tiktoken.encoding_for_model('gpt-3.5-turbo'),
};

const keyHashCache: Record<string, string> = {};
const keyHash = (key: string) => {
  if (keyHashCache[key]) return keyHashCache[key];

  return (keyHashCache[key] = createHash('md5').update(key).digest('hex'));
};

export interface ChatGTPResponse {
  response: string;
  tokenUsage: {
    prompt: number;
    completion: number;
    total: number;
  };
}

export type ChatGptPromptMessages =
  OpenAI.Chat.ChatCompletionCreateParamsNonStreaming['messages'];

type OpenAICredentials = {
  type: 'openai';
  keys: string[];
};

type OpenAIAzureCredentials = {
  type: 'openai-azure';
  endpoint: string;
  version: string;
  key: string;
};

export type AICredentials = OpenAICredentials | OpenAIAzureCredentials;

function getOpenAiClient(credentials: AICredentials): OpenAI {
  switch (credentials.type) {
    case 'openai': {
      const { keys } = credentials;

      // Select random key from the given list of keys
      const randomKeyIdx = Math.floor(Math.random() * keys.length);
      const selectedKey = keys[randomKeyIdx];

      return new OpenAI({ apiKey: selectedKey });
    }
    case 'openai-azure': {
      return new AzureOpenAI({
        endpoint: credentials.endpoint,
        apiKey: credentials.key,
        apiVersion: credentials.version,
      });
    }
  }
}

@Injectable()
export class OpenaiService {
  private readonly logger: Logger;
  private readonly rateLimiter: RateLimiterMemory;
  private readonly embedRateLimiter: RateLimiterMemory;
  private readonly defaultCredentials: AICredentials;

  constructor(private appConfig: AppConfigService) {
    switch (this.appConfig.get('aiProvider')) {
      case 'openai':
        this.defaultCredentials = {
          type: 'openai',
          keys: [
            this.appConfig.get('openaiKey'),
            this.appConfig.get('openaiKey2'),
          ],
        };
        break;
      case 'openai-azure':
        this.defaultCredentials = {
          type: 'openai-azure',
          endpoint: this.appConfig.get('openaiAzureEndpoint'),
          key: this.appConfig.get('openaiAzureKey'),
          version: this.appConfig.get('openaiAzureVersion'),
        };
        break;
    }

    this.logger = new Logger(OpenaiService.name);

    // Rate limiter for 100 req / min
    this.embedRateLimiter = new RateLimiterMemory({
      points: 400,
      duration: 60 * 1,
    });

    this.rateLimiter = new RateLimiterMemory({
      points: 600,
      duration: 60 * 1,
    });
  }

  getTokenCount(input: string): number {
    const encoder = TOKENIZERS['chatgtp'];
    const tokens = encoder.encode(input);
    return tokens.length;
  }

  /**
   * Get embedding for given string
   * @param input
   * @returns
   */
  async getEmbedding(
    input: string,
    credentials?: AICredentials,
    model: EmbeddingModel = EmbeddingModel.OPENAI_EMBEDDING_2,
  ): Promise<number[] | undefined> {
    // Get openAi client from the given keys
    credentials = credentials || this.defaultCredentials;
    const openAiClient = getOpenAiClient(credentials);

    // Rate limiter check
    try {
      await this.embedRateLimiter.consume(
        `openai-emd-${keyHash(openAiClient.apiKey)}`,
        1,
      );
    } catch (err) {
      this.logger.error('OpenAI Embedding Request exceeded rate limiting');
      throw new Error('Requests exceeded maximum rate');
    }

    // API Call
    try {
      const res = await openAiClient.embeddings.create({
        input,
        model: model,
      });
      console.log('_________________ input', input);

      return res.data?.[0].embedding;
    } catch (err) {
      this.logger.error('OpenAI Embedding API error', err);
      this.logger.error('Error reponse', err?.response?.data);
      throw err;
    }
  }

  /**
 * Categorize the user's question into "Package Status", "Shops", or "General Info".
 * @param userContent The user's question as a string.
 * @param openAiClient An instance of the OpenAI client.
 * @returns The category as a string.
 */
  async getCategory(
    userContent: string,
    openAiClient: any,
  ): Promise<string> {
    try {
      const response = await openAiClient.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: `
              You are a categorization assistant. Your job is to categorize user questions into one of these three categories:
              - "Package Status"
              - "Shops"
              - "General Info"

              Rules:
              - If the user is asking about the status of their package, if they have not received their package, if the package is delayed, or if they want to track their package, respond with "Package Status".
              - If the user is asking about nearby shops or shop addresses, respond with "Shops".
              - For all other cases, respond with "General Info".

              Your response should only contain the category name and nothing else.
            `,
          },
          {
            role: 'user',
            content: userContent,
          },
        ],
        temperature: 0, // Ensure deterministic output
      });

      return response.choices[0].message.content.trim(); // Extract and return the category
    } catch (error) {
      console.error('Error in getCategory function:', error);
      throw error;
    }
  };

  /**
   * Get completions from ChatGTP
   * @param data
   * @returns
   */
  async getChatGptCompletion(
    data: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming,
    credentials?: AICredentials,
  ): Promise<ChatGTPResponse> {
    // let category = '';
    // Get openAi client from the given keys
    credentials = credentials || this.defaultCredentials;
    const openAiClient = getOpenAiClient(credentials);

    // Rate limiter check
    try {
      await this.rateLimiter.consume(
        `openai-req-${keyHash(openAiClient.apiKey)}`,
        1,
      );
    } catch (err) {
      this.logger.error('OpenAI ChatCompletion Request exeeced rate limiting');
      throw new Error('Requests exceeded maximum rate');
    }

    const userContent = Array.isArray(data.messages[0].content)
      ? data.messages[0].content.join(' ') // Combine array elements into a string
      : data.messages[0].content; 
    
    let category = await this.getCategory(userContent, openAiClient);
    console.log('_________________ category', category);
    

    // API Call
    try {
      const res = await openAiClient.chat.completions.create(data);
      let chatResponse = res.choices[0].message.content;
      if (category === 'Package Status') {
        chatResponse = 'Please provide your tracking number so I can check the status of your package.';
      } else if (category === 'Shops') { 
        chatResponse = 'Please provide your address so I can find the nearest shops for you.';
      } 

      console.log('_________________ response', chatResponse);
      
      return {
        response: chatResponse,
        tokenUsage: {
          prompt: res.usage?.prompt_tokens,
          completion: res.usage?.completion_tokens,
          total: res.usage?.total_tokens,
        },
      };
    } catch (err) {
      this.logger.error('OpenAI ChatCompletion API error', err);
      this.logger.error('Error reponse', err?.response?.data);
      throw err;
    }
  }



  /**
   * Get streaming response from chatgpt
   * @param data
   * @param completeCb
   * @returns
   */
  async getChatGptCompletionStream(
    data: OpenAI.Chat.ChatCompletionCreateParamsStreaming,
    completeCb?: (
      answer: string,
      usage: ChatGTPResponse['tokenUsage'],
    ) => Promise<void>,
    credentials?: AICredentials,
  ) {

   
    
    // Get openAi client from the given keys
    credentials = credentials || this.defaultCredentials;
    const openAiClient = getOpenAiClient(credentials);

    // Rate limiter check
    try {
      await this.rateLimiter.consume(
        `openai-req-${keyHash(openAiClient.apiKey)}`,
        1,
      );
    } catch (err) {
      this.logger.error('OpenAI ChatCompletion Request exeeced rate limiting');
      throw new Error('Requests exceeded maximum rate');
    }

    const observable = new Subject<string>();
    const promptTokens = this.getTokenCount(
      data.messages.map((m) => m.content).join(' '),
    );

    try {
      const completionStream = await openAiClient.chat.completions.create(data);

      let answer = '';

      const streamPromise = new Promise(async (res) => {
        for await (const part of completionStream) {
          const { content } = part.choices[0].delta;

          if (content !== undefined) {
            observable.next(JSON.stringify({ content }));
            answer += content;
          }
        }

        res(true);
      });

      streamPromise.then(() => {
        observable.next('[DONE]');
        observable.complete();
        const completionTokens = this.getTokenCount(answer);
        completeCb?.(answer, {
          prompt: promptTokens,
          completion: completionTokens,
          total: promptTokens + completionTokens,
        });
      });
    } catch (error) {
      if (APIError.isPrototypeOf(error)) {
        this.logger.error('OpenAI ChatCompletion API error', error);
        this.logger.error('Error response', error.data);
      }
      throw error;
    }
    console.log('_________________ I believe this is the displayed answer', observable);
    return observable;
  }
}
