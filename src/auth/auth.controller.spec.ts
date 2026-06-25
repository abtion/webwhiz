import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';

describe('AuthController', () => {
  let controller: AuthController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [AuthController],
      providers: [
        {
          provide: AuthService,
          useValue: {
            getJwtTokenForUser: jest.fn(),
            signup: jest.fn(),
            googleAuth: jest.fn(),
            getJwtTokenForUserAdmin: jest.fn(),
          },
        },
        {
          provide: ConfigService,
          useValue: {
            get: jest.fn((key: string) => {
              if (key === 'allowPublicSignup') return true;
              return undefined;
            }),
          },
        },
      ],
    }).compile();

    controller = module.get<AuthController>(AuthController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('signup', () => {
    it('should throw ForbiddenException when public signup is disabled', async () => {
      const configService = { get: jest.fn().mockReturnValue(false) } as any;
      const authService = { signup: jest.fn() } as any;
      const ctrl = new AuthController(authService, configService);

      await expect(
        ctrl.signup({ email: 'test@test.com', password: 'pass123' } as any),
      ).rejects.toThrow('Signups are disabled');
    });

    it('should call authService.signup when public signup is enabled', async () => {
      const configService = { get: jest.fn().mockReturnValue(true) } as any;
      const authService = {
        signup: jest.fn().mockResolvedValue({ id: '1' }),
      } as any;
      const ctrl = new AuthController(authService, configService);

      const result = await ctrl.signup({
        email: 'test@test.com',
        password: 'pass123',
      } as any);

      expect(authService.signup).toHaveBeenCalled();
      expect(result).toEqual({ id: '1' });
    });
  });
});
