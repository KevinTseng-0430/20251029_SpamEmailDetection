# Project Context

## Purpose
This AIoT (Artificial Intelligence of Things) project aims to integrate AI capabilities with IoT devices to create intelligent, connected systems. The project focuses on homework assignment #3, exploring practical applications of AIoT concepts.

## Tech Stack
- Python for AI/ML components
- Node.js for backend services
- MQTT for IoT communication
- TensorFlow/PyTorch for AI models
- Docker for containerization
 - scikit-learn, pandas, numpy for classical ML and data processing

## Project Conventions

### Code Style
- Python: Follow PEP 8 style guide
- JavaScript: Use ESLint with Airbnb style guide
- Use meaningful variable names that reflect their purpose
- Document all functions with docstrings/JSDoc
- Maximum line length: 100 characters

### Architecture Patterns
- Microservices architecture for scalability
- Event-driven communication using MQTT
- RESTful APIs for service interactions
- Clean architecture with clear separation of concerns
- Repository pattern for data access

### Testing Strategy
- Unit tests for all business logic
- Integration tests for API endpoints
- E2E tests for critical workflows
- Coverage requirement: minimum 80%
- Use pytest for Python, Jest for JavaScript

### Git Workflow
- Feature branch workflow
- Branch naming: feature/, bugfix/, hotfix/
- Conventional commits (feat:, fix:, docs:, etc.)
- Squash merging to main branch
- Pull request reviews required

## Domain Context
- IoT devices send telemetry data via MQTT
- AI models process real-time sensor data
- Edge computing for low-latency responses
- Device management and provisioning
- Data collection and analysis pipelines

## Important Constraints
- Low latency requirements for real-time processing
- Resource constraints on edge devices
- Data privacy and security compliance
- Network reliability and bandwidth limitations
- Power consumption optimization

## External Dependencies
- MQTT broker (e.g., Mosquitto)
- Time series database (e.g., InfluxDB)
- AI model training infrastructure
- IoT device management platform
- Monitoring and alerting system
 - Data science libraries: scikit-learn, pandas, numpy
 - Model serving / inference runtime (e.g., a lightweight Flask/FastAPI service or TF Serving)
