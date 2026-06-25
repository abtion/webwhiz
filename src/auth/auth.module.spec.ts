import { Test } from '@nestjs/testing';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';

/**
 * This test verifies that AuthController's ConfigService dependency
 * is resolvable when the module is configured like AuthModule.
 * Reproduces: "Nest can't resolve dependencies of the AuthController (AuthService, ?)"
 */
describe('AuthModule - DI resolution', () => {
  it('should resolve AuthController when ConfigModule is imported', async () => {
    const module = await Test.createTestingModule({
      imports: [ConfigModule.forRoot()],
      controllers: [AuthController],
      providers: [
        {
          provide: AuthService,
          useValue: {},
        },
      ],
    }).compile();

    const controller = module.get<AuthController>(AuthController);
    expect(controller).toBeDefined();

    const configService = module.get<ConfigService>(ConfigService);
    expect(configService).toBeDefined();
  });

  it('should fail to resolve AuthController without ConfigModule', async () => {
    await expect(
      Test.createTestingModule({
        controllers: [AuthController],
        providers: [
          {
            provide: AuthService,
            useValue: {},
          },
        ],
      }).compile(),
    ).rejects.toThrow(/ConfigService/);
  });
});
