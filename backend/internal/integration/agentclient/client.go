package agentclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type CitationItem struct {
	Key   string `json:"key"`
	Label string `json:"label"`
	Value any    `json:"value"`
}

type HistoryItem struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type AskRequest struct {
	DatasetID uint64         `json:"dataset_id"`
	SessionID uint64         `json:"session_id"`
	Question  string         `json:"question"`
	History   []HistoryItem  `json:"history"`
	Context   map[string]any `json:"context"`
}

type AskResponse struct {
	Answer      string         `json:"answer"`
	Citations   []CitationItem `json:"citations"`
	Actions     []string       `json:"actions"`
	Degraded    bool           `json:"degraded"`
	ErrorReason *string        `json:"error_reason"`
}

type Client interface {
	Health(ctx context.Context) error
	Ask(ctx context.Context, request AskRequest) (*AskResponse, error)
}

type StubClient struct{}

type HTTPClient struct {
	baseURL string
	client  *http.Client
}

func NewStubClient() Client {
	return &StubClient{}
}

func NewHTTPClient(baseURL string, timeout time.Duration) Client {
	return &HTTPClient{
		baseURL: strings.TrimRight(strings.TrimSpace(baseURL), "/"),
		client: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c *StubClient) Health(_ context.Context) error {
	return nil
}

func (c *StubClient) Ask(_ context.Context, request AskRequest) (*AskResponse, error) {
	answer := "当前智能问答已自动降级，我将基于已有分析结果给出建议。"
	if strings.TrimSpace(request.Question) != "" {
		answer = answer + "问题：" + strings.TrimSpace(request.Question)
	}

	return &AskResponse{
		Answer: answer,
		Citations: []CitationItem{
			{
				Key:   "stub_mode",
				Label: "服务模式",
				Value: "stub",
			},
		},
		Actions:  []string{"补全 Python 智能体服务配置", "检查 LLM 参数与上下文组装逻辑"},
		Degraded: true,
	}, nil
}

func (c *HTTPClient) Health(ctx context.Context) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/internal/agent/v1/health", nil)
	if err != nil {
		return err
	}
	response, err := c.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		return fmt.Errorf("智能体服务健康检查失败: status=%d body=%s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}

func (c *HTTPClient) Ask(ctx context.Context, request AskRequest) (*AskResponse, error) {
	var result AskResponse
	if err := c.doJSON(ctx, http.MethodPost, "/internal/agent/v1/ask", request, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *HTTPClient) doJSON(ctx context.Context, method, path string, payload any, target any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := c.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	if response.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return fmt.Errorf("智能体服务请求失败: status=%d body=%s", response.StatusCode, strings.TrimSpace(string(raw)))
	}
	return json.NewDecoder(response.Body).Decode(target)
}
