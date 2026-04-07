package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"live/internal/liveapp"
)

func main() {
	port := getenv("LIVE_PORT", "8090")
	dataPath := getenv("LIVE_DATA_PATH", "data/live_sample.csv")
	webPath := getenv("LIVE_WEB_PATH", "web")
	modelServiceBaseURL := getenv("LIVE_MODEL_SERVICE_BASE_URL", "http://127.0.0.1:8001")
	agentServiceBaseURL := getenv("LIVE_AGENT_SERVICE_BASE_URL", modelServiceBaseURL)
	forecastModelType := getenv("LIVE_FORECAST_MODEL_TYPE", "transformer")

	serviceClient := liveapp.NewServiceClient(
		modelServiceBaseURL,
		agentServiceBaseURL,
		forecastModelType,
		15*time.Second,
	)

	simulator, err := liveapp.NewSimulator(dataPath, time.Second, serviceClient)
	if err != nil {
		log.Fatalf("初始化 live 模拟器失败: %v", err)
	}
	simulator.Start()

	server := liveapp.NewServer(simulator, webPath)
	addr := ":" + port
	log.Printf("live 模块启动完成: http://127.0.0.1%s", addr)

	if err := http.ListenAndServe(addr, server.Handler()); err != nil {
		log.Fatalf("启动 HTTP 服务失败: %v", err)
	}
}

func getenv(key string, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
