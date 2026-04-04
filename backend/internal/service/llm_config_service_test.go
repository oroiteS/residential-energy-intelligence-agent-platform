package service

import (
	"testing"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

func TestValidateLLMConfigPayloadRejectsInvalidTemperature(t *testing.T) {
	appErr := validateLLMConfigPayload(llmConfigPayloadForTest(2.5, 60))
	if appErr == nil {
		t.Fatal("validateLLMConfigPayload() 预期返回错误，实际为 nil")
	}
	if appErr.Message != "temperature 必须在 0 到 2 之间" {
		t.Fatalf("unexpected message: %s", appErr.Message)
	}
}

func TestValidateLLMConfigPayloadRejectsInvalidTimeout(t *testing.T) {
	appErr := validateLLMConfigPayload(llmConfigPayloadForTest(0.5, 0))
	if appErr == nil {
		t.Fatal("validateLLMConfigPayload() 预期返回错误，实际为 nil")
	}
	if appErr.Message != "timeout_seconds 必须大于 0" {
		t.Fatalf("unexpected message: %s", appErr.Message)
	}
}

func llmConfigPayloadForTest(temperature float64, timeoutSeconds int) domain.LLMConfigPayload {
	return domain.LLMConfigPayload{
		Name:           "deepseek-local",
		BaseURL:        "https://example.com/v1",
		APIKey:         "sk-test",
		ModelName:      "deepseek-chat",
		Temperature:    temperature,
		TimeoutSeconds: timeoutSeconds,
		IsDefault:      true,
	}
}
