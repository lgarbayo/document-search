# Changelog

All notable changes to AtopAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Advanced date range filtering (between operations)
- Multi-provider LLM dynamic switching
- Document analytics dashboard
- CLIP-based image search integration

## [1.1.0] - 2026-03-10

### AtopAI Rebranding & Student Focus
#### Added
- 🚀 **New Mission**: Project pivoted to focus on student productivity and daily academic use.
- 🌐 **Live URL**: Integrated `atopai.cloud` as the official live deployment.
- 🇬🇧 **Internationalization**: Full translation of project documentation to English.

#### Features
- **Semantic Search**: Context-aware retrieval for accurate document querying.
- **RAG Chat**: Real-time interaction with aggregated document knowledge.
- **AI Synthesis**: Summarization and explanation of complex academic materials.
- **Improved UI**: Sleek, student-centric design focused on usability.

#### Infrastructure
- **Reverse Proxy**: Integrated **Caddy** for automatic HTTPS and efficient request routing.
- **AWS Deployment**: Automated pipeline for continuous deployment to EC2.
- **Docker Orchestration**: Refined `docker-compose` setup including Qdrant, Redis, and Celery.
- **Scalability**: Optimized background processing for heavy document uploads.

## [1.0.0] - 2026-03-01

### Initial Hackathon Release (MeigaSearch)

#### Added
- ✨ **Magic Ingestion**: Supports PDF, XLSX, CSV, TXT, PNG, JPG with automatic OCR.
- 🔍 **Hybrid Search**: Concurrent semantic and lexical search without external dependencies.
- 🤖 **RAG Co-pilot**: Automatic query expansion using local LLMs.
- 📁 **Advanced Filters**: Filter by month, year, author, category, and file type.
- 🛡️ **JWT Authentication**: RBAC-based access control.
- ⚡ **Async Processing**: Celery + Redis for non-blocking indexing.
- 🗂️ **Vector Database**: Qdrant for optimized embedding storage.

#### Infrastructure (Initial)
- Docker Compose orchestration for all core services.
- Database initialization scripts.
- Hot-reload support for development environments.
