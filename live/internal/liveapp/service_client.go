package liveapp

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

const (
	defaultLiveDatasetID = 1
	defaultLiveSessionID = 1
)

type chatHistoryItem struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ServiceClient struct {
	httpClient        *http.Client
	modelBaseURL      string
	agentBaseURL      string
	forecastModelType string
	datasetID         int
	sessionID         int
}

type predictClassificationRequest struct {
	ModelType string               `json:"model_type"`
	DatasetID int                  `json:"dataset_id"`
	Window    requestWindow        `json:"window"`
	Series    []serviceSeriesPoint `json:"series"`
	Metadata  requestMetadata      `json:"metadata"`
}

type predictClassificationResponse struct {
	ModelType      string `json:"model_type"`
	Date           string `json:"date"`
	PredictedLabel string `json:"predicted_label"`
}

type forecastRequest struct {
	ModelType     string               `json:"model_type"`
	DatasetID     int                  `json:"dataset_id"`
	ForecastStart string               `json:"forecast_start"`
	ForecastEnd   string               `json:"forecast_end"`
	Granularity   string               `json:"granularity"`
	Series        []serviceSeriesPoint `json:"series"`
	Metadata      requestMetadata      `json:"metadata"`
}

type forecastResponse struct {
	ModelType   string    `json:"model_type"`
	Predictions []float64 `json:"predictions"`
}

type agentAskRequest struct {
	DatasetID int               `json:"dataset_id"`
	SessionID int               `json:"session_id"`
	Question  string            `json:"question"`
	History   []chatHistoryItem `json:"history"`
	Context   map[string]any    `json:"context"`
}

type agentAskResponse struct {
	Answer      string `json:"answer"`
	Degraded    bool   `json:"degraded"`
	ErrorReason string `json:"error_reason"`
}

type requestWindow struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

type requestMetadata struct {
	Granularity string `json:"granularity"`
	Unit        string `json:"unit"`
}

type serviceSeriesPoint struct {
	Timestamp            string  `json:"timestamp"`
	Aggregate            float64 `json:"aggregate"`
	ActiveApplianceCount int     `json:"active_appliance_count"`
	BurstEventCount      int     `json:"burst_event_count"`
}

func NewServiceClient(
	modelBaseURL string,
	agentBaseURL string,
	forecastModelType string,
	timeout time.Duration,
) *ServiceClient {
	if timeout <= 0 {
		timeout = 15 * time.Second
	}

	normalizedForecastModelType := strings.TrimSpace(strings.ToLower(forecastModelType))
	if normalizedForecastModelType == "" {
		normalizedForecastModelType = "transformer"
	}

	return &ServiceClient{
		httpClient:        &http.Client{Timeout: timeout},
		modelBaseURL:      strings.TrimRight(strings.TrimSpace(modelBaseURL), "/"),
		agentBaseURL:      strings.TrimRight(strings.TrimSpace(agentBaseURL), "/"),
		forecastModelType: normalizedForecastModelType,
		datasetID:         defaultLiveDatasetID,
		sessionID:         defaultLiveSessionID,
	}
}

func (c *ServiceClient) PredictClassification(ctx context.Context, day daySeries) (*predictClassificationResponse, error) {
	if c == nil || c.modelBaseURL == "" {
		return nil, fmt.Errorf("未配置 LIVE_MODEL_SERVICE_BASE_URL")
	}

	request := predictClassificationRequest{
		ModelType: "tcn",
		DatasetID: c.datasetID,
		Window: requestWindow{
			Start: toRFC3339(day.Points[0].Timestamp),
			End:   toRFC3339(day.Points[len(day.Points)-1].Timestamp),
		},
		Series: buildServiceSeries(day.Points),
		Metadata: requestMetadata{
			Granularity: "15min",
			Unit:        "w",
		},
	}

	var response predictClassificationResponse
	if err := c.doJSON(ctx, c.modelBaseURL, "/internal/model/v1/predict", request, &response); err != nil {
		return nil, err
	}
	return &response, nil
}

func (c *ServiceClient) Forecast(ctx context.Context, historyPoints []DataPoint, targetDay daySeries) (*forecastResponse, error) {
	if c == nil || c.modelBaseURL == "" {
		return nil, fmt.Errorf("未配置 LIVE_MODEL_SERVICE_BASE_URL")
	}

	request := forecastRequest{
		ModelType:     c.forecastModelType,
		DatasetID:     c.datasetID,
		ForecastStart: toRFC3339(targetDay.Points[0].Timestamp),
		ForecastEnd:   toRFC3339(targetDay.Points[len(targetDay.Points)-1].Timestamp),
		Granularity:   "15min",
		Series:        buildServiceSeries(historyPoints),
		Metadata: requestMetadata{
			Granularity: "15min",
			Unit:        "w",
		},
	}

	var response forecastResponse
	if err := c.doJSON(ctx, c.modelBaseURL, "/internal/model/v1/forecast", request, &response); err != nil {
		return nil, err
	}
	return &response, nil
}

func (c *ServiceClient) Ask(
	ctx context.Context,
	question string,
	history []chatHistoryItem,
	contextPayload map[string]any,
) (*agentAskResponse, error) {
	if c == nil || c.agentBaseURL == "" {
		return nil, fmt.Errorf("未配置 LIVE_AGENT_SERVICE_BASE_URL")
	}

	request := agentAskRequest{
		DatasetID: c.datasetID,
		SessionID: c.sessionID,
		Question:  strings.TrimSpace(question),
		History:   append([]chatHistoryItem(nil), history...),
		Context:   contextPayload,
	}

	var response agentAskResponse
	if err := c.doJSON(ctx, c.agentBaseURL, "/internal/agent/v1/ask", request, &response); err != nil {
		return nil, err
	}
	return &response, nil
}

func (c *ServiceClient) doJSON(
	ctx context.Context,
	baseURL string,
	path string,
	payload any,
	target any,
) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		baseURL+path,
		bytes.NewReader(body),
	)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := c.httpClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	if response.StatusCode >= http.StatusMultipleChoices {
		raw, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return fmt.Errorf(
			"请求失败: status=%d body=%s",
			response.StatusCode,
			strings.TrimSpace(string(raw)),
		)
	}
	return json.NewDecoder(response.Body).Decode(target)
}

func buildServiceSeries(points []DataPoint) []serviceSeriesPoint {
	series := make([]serviceSeriesPoint, 0, len(points))
	for _, point := range points {
		series = append(series, serviceSeriesPoint{
			Timestamp:            toRFC3339(point.Timestamp),
			Aggregate:            point.Aggregate,
			ActiveApplianceCount: int(point.ActiveApplianceCount),
			BurstEventCount:      int(point.BurstEventCount),
		})
	}
	return series
}

func toRFC3339(value string) string {
	parsedTime, err := time.ParseInLocation("2006-01-02 15:04:05", value, time.Local)
	if err != nil {
		return value
	}
	return parsedTime.Format(time.RFC3339)
}
