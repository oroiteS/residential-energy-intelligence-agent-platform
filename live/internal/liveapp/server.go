package liveapp

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"path/filepath"
)

type Server struct {
	simulator *Simulator
	staticDir string
	mux       *http.ServeMux
}

func NewServer(simulator *Simulator, staticDir string) *Server {
	server := &Server{
		simulator: simulator,
		staticDir: staticDir,
		mux:       http.NewServeMux(),
	}
	server.routes()
	return server
}

func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) routes() {
	s.mux.HandleFunc("/api/state", s.handleState)
	s.mux.HandleFunc("/api/stream", s.handleStream)
	s.mux.HandleFunc("/api/chat", s.handleChat)

	fileServer := http.FileServer(http.Dir(s.staticDir))
	s.mux.Handle("/", s.withNoCache(fileServer))
}

func (s *Server) handleState(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		writeError(writer, http.StatusMethodNotAllowed, "仅支持 GET")
		return
	}
	writeJSON(writer, http.StatusOK, s.simulator.Snapshot())
}

func (s *Server) handleStream(writer http.ResponseWriter, request *http.Request) {
	flusher, ok := writer.(http.Flusher)
	if !ok {
		writeError(writer, http.StatusInternalServerError, "当前环境不支持流式刷新")
		return
	}

	writer.Header().Set("Content-Type", "text/event-stream")
	writer.Header().Set("Cache-Control", "no-cache")
	writer.Header().Set("Connection", "keep-alive")
	writer.Header().Set("Access-Control-Allow-Origin", "*")

	updates, cancel := s.simulator.Subscribe()
	defer cancel()

	for {
		select {
		case snapshot, ok := <-updates:
			if !ok {
				return
			}
			payload, err := json.Marshal(snapshot)
			if err != nil {
				log.Printf("序列化 SSE 消息失败: %v", err)
				return
			}
			_, _ = writer.Write([]byte("event: tick\n"))
			_, _ = writer.Write([]byte("data: "))
			_, _ = writer.Write(payload)
			_, _ = writer.Write([]byte("\n\n"))
			flusher.Flush()
		case <-request.Context().Done():
			return
		}
	}
}

func (s *Server) handleChat(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		writeError(writer, http.StatusMethodNotAllowed, "仅支持 POST")
		return
	}

	var chatRequest ChatRequest
	if err := json.NewDecoder(request.Body).Decode(&chatRequest); err != nil {
		writeError(writer, http.StatusBadRequest, "请求体不是合法 JSON")
		return
	}

	response := s.simulator.Answer(chatRequest.Question)
	writeJSON(writer, http.StatusOK, response)
}

func writeJSON(writer http.ResponseWriter, statusCode int, payload any) {
	writer.Header().Set("Content-Type", "application/json; charset=utf-8")
	writer.WriteHeader(statusCode)
	if err := json.NewEncoder(writer).Encode(payload); err != nil && !errors.Is(err, http.ErrHandlerTimeout) {
		log.Printf("写入 JSON 响应失败: %v", err)
	}
}

func writeError(writer http.ResponseWriter, statusCode int, message string) {
	writeJSON(writer, statusCode, map[string]string{"error": message})
}

func (s *Server) withNoCache(next http.Handler) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method == http.MethodGet || request.Method == http.MethodHead {
			writer.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
			writer.Header().Set("Pragma", "no-cache")
			writer.Header().Set("Expires", "0")
		}
		next.ServeHTTP(writer, request)
	})
}

func ResolvePath(baseDir string, relative string) string {
	return filepath.Clean(filepath.Join(baseDir, relative))
}
