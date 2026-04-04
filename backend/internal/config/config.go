package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

type AppConfig struct {
	AppName             string
	AppHost             string
	AppPort             string
	ServerAddr          string
	GinMode             string
	MySQLDSN            string
	ModelServiceBaseURL string
	AgentServiceBaseURL string
	RequestTimeout      time.Duration
	UseStubClients      bool
	DataUploadDir       string
	OutputRootDir       string
}

func Load() (*AppConfig, error) {
	_ = godotenv.Load()

	timeoutSeconds, err := parseIntEnv("REQUEST_TIMEOUT_SECONDS", 15)
	if err != nil {
		return nil, err
	}

	cfg := &AppConfig{
		AppName:        getenv("APP_NAME", "resident-energy-backend"),
		AppHost:        getenv("APP_HOST", "127.0.0.1"),
		AppPort:        getenv("APP_PORT", "8888"),
		GinMode:        getenv("GIN_MODE", "debug"),
		MySQLDSN:       strings.TrimSpace(os.Getenv("MYSQL_DSN")),
		RequestTimeout: time.Duration(timeoutSeconds) * time.Second,
		UseStubClients: parseBoolEnv("USE_STUB_CLIENTS", true),
		DataUploadDir:  getenv("DATA_UPLOAD_DIR", "./uploads/datasets"),
		OutputRootDir:  getenv("OUTPUT_ROOT_DIR", "./outputs"),
	}
	pythonServiceBaseURL := strings.TrimSpace(os.Getenv("PYTHON_SERVICE_BASE_URL"))
	modelServiceBaseURL := strings.TrimSpace(os.Getenv("MODEL_SERVICE_BASE_URL"))
	agentServiceBaseURL := strings.TrimSpace(os.Getenv("AGENT_SERVICE_BASE_URL"))
	if modelServiceBaseURL == "" {
		modelServiceBaseURL = pythonServiceBaseURL
	}
	if agentServiceBaseURL == "" {
		if pythonServiceBaseURL != "" {
			agentServiceBaseURL = pythonServiceBaseURL
		} else {
			agentServiceBaseURL = modelServiceBaseURL
		}
	}
	cfg.ModelServiceBaseURL = modelServiceBaseURL
	cfg.AgentServiceBaseURL = agentServiceBaseURL
	cfg.ServerAddr = normalizeAddr(cfg.AppHost, cfg.AppPort)

	return cfg, nil
}

func getenv(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func parseIntEnv(key string, fallback int) (int, error) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback, nil
	}

	value, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("%s 不是合法整数: %w", key, err)
	}
	return value, nil
}

func parseBoolEnv(key string, fallback bool) bool {
	raw := strings.TrimSpace(strings.ToLower(os.Getenv(key)))
	if raw == "" {
		return fallback
	}

	switch raw {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func normalizeAddr(host, port string) string {
	normalizedHost := strings.TrimSpace(host)
	if normalizedHost == "" {
		normalizedHost = "127.0.0.1"
	}

	normalizedPort := strings.TrimSpace(port)
	if normalizedPort == "" {
		normalizedPort = "8888"
	}

	if strings.HasPrefix(normalizedPort, ":") {
		return normalizedHost + normalizedPort
	}

	return normalizedHost + ":" + normalizedPort
}
