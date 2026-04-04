export type ApiEnvelope<T> = {
  code: string
  message: string
  data: T
  request_id: string
  timestamp: string
}

export type Pagination = {
  page: number
  page_size: number
  total: number
}
