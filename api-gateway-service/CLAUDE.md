# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a unified AI services API gateway built with FastAPI that aggregates multiple AI services (STT, TTS, OCR, User management, LLM, and MD2PDF) behind a single entry point. The gateway provides OAuth2 Bearer token authentication, IP-based security features, and request forwarding to backend services.

## Architecture

### Core Components

- **Main API Gateway** (`api.py`): FastAPI application that handles authentication, request routing, and service aggregation
- **Authentication Module** (`authentication/auth.py`): JWT-based OAuth2 authentication system with bcrypt password hashing
- **Service Configuration**: Backend services are configured via `BACKEND_SERVICES` dictionary with base URLs and health check endpoints

### Backend Services Integration

The gateway proxies requests to these services:
- **LLM Service** (port 61080): Chat completions, image proxy, title generation
- **STT Service** (port 57001): Speech-to-text transcription
- **TTS Service** (port 57002): Text-to-speech synthesis  
- **OCR Service** (port 57004): Optical character recognition
- **User Service** (port 58000): User and chat session management
- **MD2PDF Service** (port 55000): Markdown to PDF conversion

### Security Features

- **OAuth2 Bearer Token Authentication**: JWT tokens with configurable expiration
- **IP Login Lockout**: Configurable failed login attempt protection (disabled by default)
- **Concurrent Request Limiting**: Per-IP request throttling (disabled by default)
- **HTTPS Support**: SSL certificate configuration for production deployment

## Development Commands

### Running the Gateway

```bash
# Development mode with auto-reload
python api.py

# Production mode (modify environment variables as needed)
GATEWAY_PORT=60443 GATEWAY_RELOAD=false LOG_LEVEL=info python api.py
```

### Environment Configuration

Key environment variables:
- `GATEWAY_PORT`: Server port (default: 60443)
- `GATEWAY_RELOAD`: Enable auto-reload for development (default: true)
- `LOG_LEVEL`: Logging level (default: info)
- `LLM_SERVICE_URL`, `STT_SERVICE_URL`, etc.: Backend service URLs
- `SSL_ROOT_DIR`, `SSL_CERT_NAME`, `SSL_KEY_NAME`: SSL certificate configuration

### Authentication Setup

1. Configure credentials in `authentication/auth_config.json`:
   ```json
   {
     "FIXED_USERNAME": "your_username",
     "FIXED_PASSWORD_HASH": "bcrypt_hash",
     "JWT_SECRET_KEY": "base64_secret"
   }
   ```

2. Generate password hash: `python authentication/generate_hash_token.py`
3. Generate JWT secret: `python authentication/generate_jwt_token.py`

### Testing

```bash
# Basic API test
python test.py

# Health check
curl -X GET "http://localhost:60443/health"

# Get auth token
curl -X POST "http://localhost:60443/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your_username&password=your_password"
```

## Code Patterns

### Request Forwarding

The `forward_request()` function handles:
- Header filtering and forwarding
- Client IP preservation via X-Forwarded-For
- Stream vs non-stream response handling
- Error handling and logging

### Service Router Pattern

Each backend service has its own APIRouter with:
- Service-specific prefix (e.g., `/stt`, `/tts`)
- Authentication dependency injection
- Specialized request handling for file uploads, form data, JSON payloads

### Middleware Architecture

- `ConcurrentRequestLimitMiddleware`: IP-based request throttling
- Global exception handling for backend service errors
- Comprehensive logging for debugging and monitoring

## Configuration Notes

- Authentication module loading is conditional - gateway runs in degraded mode if auth fails
- Backend service health checks use configurable endpoints and HTTP methods
- SSL/HTTPS is automatically disabled if certificate files are missing
- Hot reload monitors specific directories and file types in development mode