package agentclient

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type CitationItem struct {
	Key   string `json:"key"`
	Label string `json:"label"`
	Value any    `json:"value"`
}

type MissingInformationItem struct {
	Key      string `json:"key"`
	Question string `json:"question"`
	Reason   string `json:"reason"`
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
	Answer             string                   `json:"answer"`
	Citations          []CitationItem           `json:"citations"`
	Actions            []string                 `json:"actions"`
	MissingInformation []MissingInformationItem `json:"missing_information"`
	ConfidenceLevel    *string                  `json:"confidence_level"`
	Intent             string                   `json:"intent"`
	Degraded           bool                     `json:"degraded"`
	ErrorReason        *string                  `json:"error_reason"`
}

type ReportSummaryRequest struct {
	DatasetID uint64         `json:"dataset_id"`
	Context   map[string]any `json:"context"`
}

type ReportSection struct {
	Title string `json:"title"`
	Body  string `json:"body"`
}

type ReportSummaryResponse struct {
	Title           string          `json:"title"`
	Overview        string          `json:"overview"`
	Sections        []ReportSection `json:"sections"`
	Recommendations []string        `json:"recommendations"`
	Degraded        bool            `json:"degraded"`
	ErrorReason     *string         `json:"error_reason"`
}

type RenderPDFRequest struct {
	Markdown    string `json:"markdown"`
	Title       string `json:"title"`
	Author      string `json:"author"`
	Date        string `json:"date"`
	Theme       string `json:"theme"`
	Cover       bool   `json:"cover"`
	TOC         bool   `json:"toc"`
	HeaderTitle string `json:"header_title"`
	FooterLeft  string `json:"footer_left"`
}

type RenderPDFResponse struct {
	FileName    string `json:"file_name"`
	ContentType string `json:"content_type"`
	PDFBase64   string `json:"pdf_base64"`
	FileSize    int    `json:"file_size"`
}

type Client interface {
	Health(ctx context.Context) error
	Ask(ctx context.Context, request AskRequest) (*AskResponse, error)
	SummarizeReport(ctx context.Context, request ReportSummaryRequest) (*ReportSummaryResponse, error)
	RenderPDF(ctx context.Context, request RenderPDFRequest) ([]byte, error)
}

type StubClient struct{}

type HTTPClient struct {
	baseURLs []string
	client   *http.Client
}

func NewStubClient() Client {
	return &StubClient{}
}

func NewHTTPClient(baseURL string, timeout time.Duration) Client {
	return &HTTPClient{
		baseURLs: buildCandidateBaseURLs(baseURL),
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
	confidenceLevel := "low"

	return &AskResponse{
		Answer: answer,
		Citations: []CitationItem{
			{
				Key:   "stub_mode",
				Label: "服务模式",
				Value: "stub",
			},
		},
		Actions:            []string{"补全 Python 智能体服务配置", "检查 LLM 参数与上下文组装逻辑"},
		MissingInformation: []MissingInformationItem{},
		ConfidenceLevel:    &confidenceLevel,
		Intent:             "advice",
		Degraded:           true,
	}, nil
}

func (c *StubClient) SummarizeReport(_ context.Context, _ ReportSummaryRequest) (*ReportSummaryResponse, error) {
	reason := "AGENT_STUB_MODE"
	return &ReportSummaryResponse{
		Title:    "居民用电分析报告",
		Overview: "当前智能体运行在降级模式，报告内容基于现有统计分析结果自动整理。",
		Sections: []ReportSection{
			{
				Title: "执行说明",
				Body:  "已使用本地降级逻辑生成报告摘要，可正常用于 PDF 导出与归档。",
			},
		},
		Recommendations: []string{
			"补全 LLM 环境变量以启用更完整的报告总结。",
			"结合分类与预测结果复核峰时段负荷安排。",
		},
		Degraded:    true,
		ErrorReason: &reason,
	}, nil
}

func (c *StubClient) RenderPDF(_ context.Context, _ RenderPDFRequest) ([]byte, error) {
	return nil, fmt.Errorf("stub 模式不支持 PDF 渲染")
}

func (c *HTTPClient) Health(ctx context.Context) error {
	var lastErr error
	for _, baseURL := range c.baseURLs {
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/internal/agent/v1/health", nil)
		if err != nil {
			return err
		}
		response, err := c.client.Do(request)
		if err != nil {
			lastErr = err
			continue
		}
		if response.StatusCode >= 300 {
			body, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
			_ = response.Body.Close()
			return fmt.Errorf("智能体服务健康检查失败: status=%d body=%s", response.StatusCode, strings.TrimSpace(string(body)))
		}
		_ = response.Body.Close()
		return nil
	}
	if lastErr != nil {
		return lastErr
	}
	return fmt.Errorf("智能体服务基础地址未配置")
}

func (c *HTTPClient) Ask(ctx context.Context, request AskRequest) (*AskResponse, error) {
	var result AskResponse
	if err := c.doJSON(ctx, http.MethodPost, "/internal/agent/v1/ask", request, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *HTTPClient) SummarizeReport(ctx context.Context, request ReportSummaryRequest) (*ReportSummaryResponse, error) {
	var result ReportSummaryResponse
	if err := c.doJSON(ctx, http.MethodPost, "/internal/agent/v1/report-summary", request, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *HTTPClient) RenderPDF(ctx context.Context, request RenderPDFRequest) ([]byte, error) {
	var result RenderPDFResponse
	if err := c.doJSON(ctx, http.MethodPost, "/internal/agent/v1/render-pdf", request, &result); err != nil {
		return nil, err
	}
	if strings.TrimSpace(result.PDFBase64) == "" {
		return nil, fmt.Errorf("智能体服务未返回 PDF 内容")
	}
	decoded, err := base64.StdEncoding.DecodeString(result.PDFBase64)
	if err != nil {
		return nil, fmt.Errorf("解析 PDF 响应失败: %w", err)
	}
	return decoded, nil
}

func (c *HTTPClient) doJSON(ctx context.Context, method, path string, payload any, target any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	var lastErr error
	for _, baseURL := range c.baseURLs {
		request, requestErr := http.NewRequestWithContext(ctx, method, baseURL+path, bytes.NewReader(body))
		if requestErr != nil {
			return requestErr
		}
		request.Header.Set("Content-Type", "application/json")

		response, requestErr := c.client.Do(request)
		if requestErr != nil {
			lastErr = requestErr
			continue
		}

		if response.StatusCode >= 300 {
			raw, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
			_ = response.Body.Close()
			return fmt.Errorf("智能体服务请求失败: status=%d body=%s", response.StatusCode, strings.TrimSpace(string(raw)))
		}
		decodeErr := json.NewDecoder(response.Body).Decode(target)
		_ = response.Body.Close()
		return decodeErr
	}
	if lastErr != nil {
		return lastErr
	}
	return fmt.Errorf("智能体服务基础地址未配置")
}

func buildCandidateBaseURLs(baseURL string) []string {
	normalized := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if normalized == "" {
		return nil
	}

	candidates := []string{normalized}
	for _, host := range []string{"models-agent", "host.docker.internal"} {
		if fallback := replaceBaseURLHost(normalized, host, "127.0.0.1"); fallback != "" && fallback != normalized {
			candidates = append(candidates, fallback)
		}
	}
	return candidates
}

func replaceBaseURLHost(baseURL string, fromHost string, toHost string) string {
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return ""
	}
	if !strings.EqualFold(parsed.Hostname(), fromHost) {
		return ""
	}
	port := parsed.Port()
	if port == "" {
		parsed.Host = toHost
	} else {
		parsed.Host = toHost + ":" + port
	}
	return strings.TrimRight(parsed.String(), "/")
}
