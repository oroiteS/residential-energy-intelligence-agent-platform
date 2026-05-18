/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_USE_MOCK?: string
  readonly VITE_BACKEND_BASE_URL?: string
  readonly VITE_API_PREFIX?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
