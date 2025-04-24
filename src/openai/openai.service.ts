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

      return res.data?.[0].embedding;
    } catch (err) {
      this.logger.error('OpenAI Embedding API error', err);
      this.logger.error('Error reponse', err?.response?.data);
      throw err;
    }
  }

  /**
   * Categorize the user's question into "Package Status", "Shops", or "General Info".
   * @param input The user's question as a string.
   * @returns The category as a string.
   */
  async analyzeChatConversation(
    input: any,
    openAiClient: OpenAI,
  ): Promise<string> {
    const prompt = `
      The chat history is: "${input}".
      Determine whether any content in the chat history relates to package tracking or tracking numbers (e.g., questions about shipments, delivery, or tracking numbers) or address (e.g., questions about pickup address relative to user address).
      Respond with "Provide tracking number" if it is related to tracking, but does not have a number consisting of 6 or more digits, if the user have provided a number respond with the provided number.
      If any content in the chat history is related to an address, respond with the address in this format: {street:"[STREETNAME]", streetNumber:"[STREETNUMBER]" ,zip:"[ZIPCODE]"} (e.g., street:"Grækenlandsvej", streetNumber: "100",zip:"2300").
      If only the street name is provided, and the zip code was previously mentioned respond with street and zip in before mentioned address format.
      If only the zip code is provided, and the street name was previously mentioned respond with street and zip in before mentioned address format.
      `;

    try {
      const response = await openAiClient.chat.completions.create({
        model: 'gpt-4', // Or use another model
        messages: [{ role: 'system', content: prompt }],
        temperature: 0, // To make the response more deterministic
      });

      const result = response.choices[0]?.message?.content?.trim();

      return result;
    } catch (error) {
      console.error('Error analyzing user input:', error);
      throw error;
    }
  }

  getAddress(analyzedInput: string): {
    id: string;
    street: string;
    streetNumber: string;
    zip: string;
  } {
    const regex =
      /street:\s*"(.*?)"\s*,\s*streetNumber:\s*"(.*?)"\s*,\s*zip:\s*"(.*?)"/;
    const matches = regex.exec(analyzedInput);

    if (!matches) {
      return { id: 'pakkeshopData', street: '', streetNumber: '', zip: '' };
    }

    const [, street, streetNumber, zip] = matches;

    return { id: 'pakkeshopData', street, streetNumber, zip };
  }

  async fetchPakkeshopInformation(obj): Promise<any> {
    if (obj.street === '' || obj.streetNumber === '' || obj.zip === '')
      return null;

    const apiUrl = `https://api.dao.as/DAOPakkeshop/FindPakkeshop.php?kundeid=${process.env.DAO_API_URL_CUSTOMER_ID}&kode=${process.env.DAO_API_URL_CODE}&postnr=${obj.zip}&adresse=${obj.street}%${obj.streetNumber}}&format=json&antal=5`;

    try {
      const response = await fetch(apiUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching tracking information:', error);
      throw error;
    }
  }

  getTrackingNumber(analyzedInput: string): undefined | string {
    const regex = /\b(?:\d{9}|7\d{12}|00057\d{15})\b/g;
    const matches = analyzedInput.match(regex);
    if (!matches) return null;

    return matches[0];
  }

  async fetchTrackingInformation(trackingNumber: string): Promise<any> {
    if (!trackingNumber) return null;

    const apiUrl = `https://api.dao.as/TrackNTrace_v2.php?kundeid=${process.env.DAO_API_URL_CUSTOMER_ID}&kode=${process.env.DAO_API_URL_CODE}&stregkode=${trackingNumber}`;

    try {
      const response = await fetch(apiUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching tracking information:', error);
      throw error;
    }
  }

  async implementApiCalls(
    analyzedInput: string,
    data:
      | OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
      | OpenAI.Chat.ChatCompletionCreateParamsStreaming,
  ) {
    const addressObject = this.getAddress(analyzedInput);
    const pakkeshopData = await this.fetchPakkeshopInformation(addressObject);

    const trackingNumber = this.getTrackingNumber(analyzedInput);
    const apiData = await this.fetchTrackingInformation(trackingNumber);

    if (pakkeshopData && pakkeshopData.status === 'OK') {
      if (pakkeshopData.status === 'OK') {
        data.messages.push({
          content: JSON.stringify({
            pakkeshops: pakkeshopData.resultat.pakkeshops,
          }),
          role: 'system',
        });
      } else {
        data.messages.push({
          content: pakkeshopData.fejltekst,
          role: 'system',
        });
      }
    } else {
      data.messages.push({
        content: JSON.stringify(addressObject),
        role: 'system',
      });
    }

    if (apiData) {
      const {
        resultat: { afhentningssted, afsender, haendelser },
      } = apiData;

      data.messages.push({
        content: JSON.stringify({ afhentningssted, afsender, haendelser }),
        role: 'system',
      });
    }

    return data;
  }

  /**
   * Get completions from ChatGTP
   * @param data
   * @returns
   */
  async getChatGptCompletion(
    data: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming,
    credentials?: AICredentials,
  ): Promise<ChatGTPResponse> {
    // Get openAi client from the given keys
    credentials = credentials || this.defaultCredentials;
    const openAiClient = getOpenAiClient(credentials);

    const analyzedInput = await this.analyzeChatConversation(
      JSON.stringify(data.messages.slice(1)),
      openAiClient,
    );

    await this.implementApiCalls(analyzedInput, data);

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

    // API Call
    try {
      const res = await openAiClient.chat.completions.create(data);
      const chatResponse = res.choices[0].message.content;

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

    const analyzedInput = await this.analyzeChatConversation(
      JSON.stringify(data.messages.slice(1)),
      openAiClient,
    );

    await this.implementApiCalls(analyzedInput, data);

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
    return observable;
  }
}
