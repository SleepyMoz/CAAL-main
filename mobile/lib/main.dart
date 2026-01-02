import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'app.dart';
import 'services/config_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Hide status bar and navigation bar for full-screen experience
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);

  // Initialize config service first
  final configService = ConfigService();
  await configService.init();

  // Try to load .env as fallback for development (optional)
  try {
    await dotenv.load(fileName: '.env');
    // If .env exists and config isn't set, migrate values
    if (!configService.isConfigured) {
      final envUrl = dotenv.env['CAAL_SERVER_URL']?.replaceAll('"', '');
      if (envUrl != null && envUrl.isNotEmpty) {
        await configService.setServerUrl(envUrl);
      }
      final envKey = dotenv.env['PORCUPINE_ACCESS_KEY']?.replaceAll('"', '');
      if (envKey != null && envKey.isNotEmpty) {
        await configService.setPorcupineAccessKey(envKey);
      }
    }
  } catch (_) {
    // .env file not found - that's fine, we'll use ConfigService
  }

  runApp(CaalApp(configService: configService));
}
