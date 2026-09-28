import { firstValueFrom, toArray } from 'rxjs';
import { AzureOpenAI } from 'openai';
import { OpenaiService } from './openai.service';
import { AppConfigService } from '../common/config/appConfig.service';

jest.mock('openai', () => ({
  __esModule: true,
  default: jest.fn(),
  AzureOpenAI: jest.fn(),
}));

const CONFIG: Record<string, string> = {
  aiProvider: 'openai-azure',
  openaiAzureEndpoint: 'https://example.openai.azure.com',
  openaiAzureKey: 'key',
  openaiAzureVersion: '2024-06-01',
};

const contentChunk = (content: string) => ({
  choices: [{ index: 0, finish_reason: null, delta: { content } }],
});

async function* streamOf(chunks: unknown[], error?: Error) {
  for (const chunk of chunks) yield chunk;
  if (error) throw error;
}

describe('OpenaiService', () => {
  let service: OpenaiService;
  let create: jest.Mock;

  beforeEach(() => {
    create = jest.fn();
    (AzureOpenAI as unknown as jest.Mock).mockImplementation(() => ({
      apiKey: 'key',
      chat: { completions: { create } },
    }));

    service = new OpenaiService({
      get: (key: string) => CONFIG[key],
    } as unknown as AppConfigService);

    jest.spyOn(service, 'analyzeChatConversation').mockResolvedValue('');
    jest.spyOn(service, 'implementApiCalls').mockResolvedValue(undefined);
    jest.spyOn(service['logger'], 'error').mockImplementation(() => undefined);
  });

  const requestStream = (completeCb?: jest.Mock) =>
    service.getChatGptCompletionStream(
      {
        messages: [{ role: 'user', content: 'Hi' }],
        model: 'gpt-4o',
        stream: true,
      },
      completeCb,
    );

  describe('getChatGptCompletion', () => {
    it.each([
      ['no choices', []],
      [
        'a filtered choice without message',
        [{ index: 0, finish_reason: 'content_filter' }],
      ],
      [
        'a null message content',
        [{ index: 0, message: { role: 'assistant', content: null } }],
      ],
    ])('returns an empty response for %s', async (_, choices) => {
      create.mockResolvedValue({ choices, usage: undefined });

      const result = await service.getChatGptCompletion({
        messages: [{ role: 'user', content: 'Hi' }],
        model: 'gpt-4o',
      });

      expect(result.response).toBe('');
    });
  });

  describe('getChatGptCompletionStream', () => {
    it('skips chunks without delta, such as Azure content filter chunks', async () => {
      create.mockResolvedValue(
        streamOf([
          { choices: [] },
          {
            choices: [
              { index: 0, finish_reason: null, content_filter_results: {} },
            ],
          },
          contentChunk('Hello'),
          { choices: [{ index: 0, finish_reason: null, delta: {} }] },
          contentChunk(' world'),
          { choices: [{ index: 0, finish_reason: 'stop', delta: {} }] },
        ]),
      );
      const completeCb = jest.fn().mockResolvedValue(undefined);

      const observable = await requestStream(completeCb);
      const values = await firstValueFrom(observable.pipe(toArray()));

      expect(values).toEqual([
        JSON.stringify({ content: 'Hello' }),
        JSON.stringify({ content: ' world' }),
        '[DONE]',
      ]);
      expect(completeCb).toHaveBeenCalledWith(
        'Hello world',
        expect.objectContaining({ prompt: expect.any(Number) }),
      );
    });

    it('errors the observable instead of rejecting when the stream fails', async () => {
      create.mockResolvedValue(
        streamOf([contentChunk('Hel')], new Error('connection reset')),
      );
      const completeCb = jest.fn();

      const observable = await requestStream(completeCb);

      await expect(firstValueFrom(observable.pipe(toArray()))).rejects.toThrow(
        'Failed to generate answer',
      );
      expect(completeCb).not.toHaveBeenCalled();
    });

    it('logs instead of rejecting when the completion callback fails', async () => {
      create.mockResolvedValue(streamOf([contentChunk('Hi')]));
      const completeCb = jest.fn().mockRejectedValue(new Error('db down'));

      const observable = await requestStream(completeCb);
      await firstValueFrom(observable.pipe(toArray()));
      await new Promise(process.nextTick);

      expect(service['logger'].error).toHaveBeenCalledWith(
        expect.stringContaining('callback'),
        expect.any(Error),
      );
    });
  });
});
