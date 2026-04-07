package modelclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type TimeSeriesPoint struct {
	Timestamp            string  `json:"timestamp"`
	Aggregate            float64 `json:"aggregate"`
	ActiveApplianceCount int     `json:"active_appliance_count"`
	BurstEventCount      int     `json:"burst_event_count"`
}

type Metadata struct {
	Granularity string `json:"granularity,omitempty"`
	Unit        string `json:"unit,omitempty"`
}

type Window struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

type PredictClassificationRequest struct {
	ModelType string            `json:"model_type"`
	DatasetID uint64            `json:"dataset_id"`
	Window    Window            `json:"window"`
	Series    []TimeSeriesPoint `json:"series"`
	Metadata  Metadata          `json:"metadata"`
}

type PredictClassificationResponse struct {
	ModelType           string  `json:"model_type"`
	SampleID            string  `json:"sample_id"`
	HouseID             string  `json:"house_id"`
	Date                string  `json:"date"`
	PredictedLabel      string  `json:"predicted_label"`
	Confidence          float64 `json:"confidence"`
	ProbDayHighNightLow float64 `json:"prob_day_high_night_low"`
	ProbDayLowNightHigh float64 `json:"prob_day_low_night_high"`
	ProbAllDayHigh      float64 `json:"prob_all_day_high"`
	ProbAllDayLow       float64 `json:"prob_all_day_low"`
	RuntimeDevice       string  `json:"runtime_device"`
	RuntimeLoss         string  `json:"runtime_loss"`
}

type ForecastRequest struct {
	ModelType     string            `json:"model_type"`
	DatasetID     uint64            `json:"dataset_id"`
	ForecastStart string            `json:"forecast_start"`
	ForecastEnd   string            `json:"forecast_end"`
	Granularity   string            `json:"granularity"`
	Series        []TimeSeriesPoint `json:"series"`
	Metadata      Metadata          `json:"metadata"`
}

type ForecastResponse struct {
	ModelType   string    `json:"model_type"`
	SampleID    string    `json:"sample_id"`
	HouseID     string    `json:"house_id"`
	InputStart  string    `json:"input_start"`
	InputEnd    string    `json:"input_end"`
	Predictions []float64 `json:"predictions"`
}

type Client interface {
	Health(ctx context.Context) error
	PredictClassification(ctx context.Context, request PredictClassificationRequest) (*PredictClassificationResponse, error)
	Forecast(ctx context.Context, request ForecastRequest) (*ForecastResponse, error)
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

func (c *StubClient) PredictClassification(_ context.Context, request PredictClassificationRequest) (*PredictClassificationResponse, error) {
	if len(request.Series) == 0 {
		return nil, fmt.Errorf("分类输入序列不能为空")
	}

	dayTotal := 0.0
	nightTotal := 0.0
	dayCount := 0
	nightCount := 0
	fullTotal := 0.0
	for _, point := range request.Series {
		fullTotal += point.Aggregate
		ts, err := time.Parse(time.RFC3339, point.Timestamp)
		if err != nil {
			continue
		}
		hour := ts.Hour()
		if hour >= 8 && hour < 18 {
			dayTotal += point.Aggregate
			dayCount++
		} else {
			nightTotal += point.Aggregate
			nightCount++
		}
	}

	dayMean := safeMean(dayTotal, dayCount)
	nightMean := safeMean(nightTotal, nightCount)
	fullMean := safeMean(fullTotal, len(request.Series))

	label := "all_day_low"
	switch {
	case dayMean >= nightMean*1.2:
		label = "day_high_night_low"
	case nightMean >= dayMean*1.2:
		label = "day_low_night_high"
	case fullMean >= 500:
		label = "all_day_high"
	default:
		label = "all_day_low"
	}

	probs := map[string]float64{
		"day_high_night_low": 0.03,
		"day_low_night_high": 0.03,
		"all_day_high":       0.03,
		"all_day_low":        0.03,
	}
	probs[label] = 0.91

	date := ""
	if ts, err := time.Parse(time.RFC3339, request.Window.Start); err == nil {
		date = ts.Format("2006-01-02")
	}

	return &PredictClassificationResponse{
		ModelType:           request.ModelType,
		SampleID:            fmt.Sprintf("%d_%s", request.DatasetID, date),
		HouseID:             "",
		Date:                date,
		PredictedLabel:      label,
		Confidence:          probs[label],
		ProbDayHighNightLow: probs["day_high_night_low"],
		ProbDayLowNightHigh: probs["day_low_night_high"],
		ProbAllDayHigh:      probs["all_day_high"],
		ProbAllDayLow:       probs["all_day_low"],
		RuntimeDevice:       "stub",
		RuntimeLoss:         "rule_based",
	}, nil
}

func (c *StubClient) Forecast(_ context.Context, request ForecastRequest) (*ForecastResponse, error) {
	if len(request.Series) == 0 {
		return nil, fmt.Errorf("预测输入序列不能为空")
	}

	predictions := make([]float64, 0, 96)
	if len(request.Series) >= 96 {
		start := len(request.Series) - 96
		for _, point := range request.Series[start:] {
			predictions = append(predictions, point.Aggregate)
		}
	} else {
		last := request.Series[len(request.Series)-1].Aggregate
		for len(predictions) < 96 {
			predictions = append(predictions, last)
		}
	}

	return &ForecastResponse{
		ModelType:   request.ModelType,
		SampleID:    fmt.Sprintf("%d_%s_%s", request.DatasetID, request.ForecastStart, request.ForecastEnd),
		HouseID:     "",
		InputStart:  request.Series[0].Timestamp,
		InputEnd:    request.Series[len(request.Series)-1].Timestamp,
		Predictions: predictions,
	}, nil
}

func (c *HTTPClient) Health(ctx context.Context) error {
	var lastErr error
	for _, baseURL := range c.baseURLs {
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/internal/model/v1/health", nil)
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
			return fmt.Errorf("模型服务健康检查失败: status=%d body=%s", response.StatusCode, strings.TrimSpace(string(body)))
		}
		_ = response.Body.Close()
		return nil
	}
	if lastErr != nil {
		return lastErr
	}
	return fmt.Errorf("模型服务基础地址未配置")
}

func (c *HTTPClient) PredictClassification(ctx context.Context, request PredictClassificationRequest) (*PredictClassificationResponse, error) {
	var result PredictClassificationResponse
	if err := c.doJSON(ctx, http.MethodPost, "/internal/model/v1/predict", request, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *HTTPClient) Forecast(ctx context.Context, request ForecastRequest) (*ForecastResponse, error) {
	var result ForecastResponse
	if err := c.doJSON(ctx, http.MethodPost, "/internal/model/v1/forecast", request, &result); err != nil {
		return nil, err
	}
	return &result, nil
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
			return fmt.Errorf("模型服务请求失败: status=%d body=%s", response.StatusCode, strings.TrimSpace(string(raw)))
		}
		decodeErr := json.NewDecoder(response.Body).Decode(target)
		_ = response.Body.Close()
		return decodeErr
	}
	if lastErr != nil {
		return lastErr
	}
	return fmt.Errorf("模型服务基础地址未配置")
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

func safeMean(total float64, count int) float64 {
	if count <= 0 {
		return 0
	}
	return total / float64(count)
}
