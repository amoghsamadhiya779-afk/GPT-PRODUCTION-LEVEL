# GPT Studio (Frontend)

GPT Studio is the primary, production-facing web interface for the `GPT-PRODUCTION-LEVEL` project. Built to provide a premium, dynamic, and responsive chat experience, it seamlessly communicates with the FastAPI model server.

## Purpose

This frontend provides a beautiful UI for text generation, model fine-tuning ("Teach Mode"), and web grounding. It connects to the custom-built GPT-2 inference engine, visualizing real-time SSE token streams and telemetry metrics (latency, tokens/second, KV-Cache utilization). 

When the backend server is unreachable, the application gracefully degrades to an offline mock mode, allowing UI exploration without a running engine.

## Technology Stack

- **Framework**: [Next.js 16 (App Router)](https://nextjs.org/)
- **UI Library**: [React 19](https://react.dev/)
- **Styling**: [Tailwind CSS 4](https://tailwindcss.com/)
- **Components**: [shadcn/ui](https://ui.shadcn.com/)
- **Animations**: [Framer Motion](https://www.framer.com/motion/)

## Environment Variables

The application relies on the following environment variable to locate the FastAPI backend. You can set this in a `.env.local` file:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```
*(Defaults to `http://localhost:8000` if not provided.)*

## Getting Started

### Prerequisites
Make sure you have [Node.js](https://nodejs.org/) installed.

### Installation & Development

First, install the dependencies:
```bash
npm install
# or yarn install / pnpm install
```

Then, run the development server:
```bash
npm run dev
# or yarn dev / pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application. It will attempt to connect to the FastAPI server on port 8000.

### Build for Production
To create an optimized production build:
```bash
npm run build
npm start
```

## How It Connects to the Backend

The frontend communicates exclusively over HTTP with the FastAPI server defined in `app/api.py`:
- **Generation**: Hits `POST /generate` and `POST /generate/stream` for generating responses. SSE is parsed natively using `fetch` and `TextDecoder` to animate tokens as they arrive.
- **Web Grounding**: Automatically appends the `web_search` toggle flag to generation requests. When sources are returned, they are displayed instantly.
- **Fine-Tuning**: Interfaces with `POST /finetune` to trigger LoRA background training and polls `GET /finetune/{job_id}` for progress updates in Teach Mode.
- **Health**: Polls `GET /health` to display model status and telemetry metrics on the boot sequence and dashboard.

## Deploy on Vercel

The easiest way to deploy this Next.js app is to use the [Vercel Platform](https://vercel.com/new).

1. Push your code to a GitHub repository.
2. Import the `frontend/` directory as the Root Directory in your Vercel project settings.
3. Add the `NEXT_PUBLIC_BACKEND_URL` environment variable pointing to your deployed FastAPI server.
4. Click Deploy.

For more details, check out the [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying).
