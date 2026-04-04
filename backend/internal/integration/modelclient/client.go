package modelclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
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

type BacktestRequest struct {
	ModelType     string            `json:"model_type"`
	DatasetID     uint64            `json:"dataset_id"`
	BacktestStart string            `json:"backtest_start"`
	BacktestEnd   string            `json:"backtest_end"`
	Granularity   string            `json:"granularity"`
	Series        []TimeSeriesPoint `json:"series"`
	Metadata      Metadata          `json:"metadata"`
}

type BacktestPoint struct {
	Timestamp string  `json:"timestamp"`
	Actual    float64 `json:"actual"`
	Predicted float64 `json:"predicted"`
}

type BacktestMetrics struct {
	MAE   float64 `json:"mae"`
	RMSE  float64 `json:"rmse"`
	SMAPE float64 `json:"smape"`
	WAPE  float64 `json:"wape"`
}

type BacktestResponse struct {
	ModelType     string          `json:"model_type"`
	BacktestStart string          `json:"backtest_start"`
	BacktestEnd   string          `json:"backtest_end"`
	Granularity   string          `json:"granularity"`
	Predictions   []BacktestPoint `json:"predictions"`
	Metrics       BacktestMetrics `json:"metrics"`
}

type Client interface {
	Health(ctx context.Context) error
	PredictClassification(ctx context.Context, request PredictClassificationRequest) (*PredictClassificationResponse, error)
	Forecast(ctx context.Context, request ForecastRequest) (*ForecastResponse, error)
	Backtest(ctx context.Context, request BacktestRequest) (*BacktestResponse, error)
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

func (c *StubClient) Backtest(_ context.Context, request BacktestRequest) (*BacktestResponse, error) {
	start, err := time.Parse(time.RFC3339, request.BacktestStart)
	if err != nil {
		return nil, err
	}
	end, err := time.Parse(time.RFC3339, request.BacktestEnd)
	if err != nil {
		return nil, err
	}

	actualPoints := make([]TimeSeriesPoint, 0, 96)
	historyPoints := make([]TimeSeriesPoint, 0, len(request.Series))
	for _, point := range request.Series {
		ts, parseErr := time.Parse(time.RFC3339, point.Timestamp)
		if parseErr != nil {
			continue
		}
		if !ts.Before(start) && !ts.After(end) {
			actualPoints = append(actualPoints, point)
			continue
		}
		if ts.Before(start) {
			historyPoints = append(historyPoints, point)
		}
	}
	if len(actualPoints) == 0 {
		return nil, fmt.Errorf("回测缺少实际值区间")
	}

	forecastResp, err := c.Forecast(context.Background(), ForecastRequest{
		ModelType:     request.ModelType,
		DatasetID:     request.DatasetID,
		ForecastStart: request.BacktestStart,
		ForecastEnd:   request.BacktestEnd,
		Granularity:   request.Granularity,
		Series:        historyPoints,
		Metadata:      request.Metadata,
	})
	if err != nil {
		return nil, err
	}

	points := make([]BacktestPoint, 0, len(actualPoints))
	actualValues := make([]float64, 0, len(actualPoints))
	predictedValues := make([]float64, 0, len(actualPoints))
	for index, actual := range actualPoints {
		predicted := actual.Aggregate
		if index < len(forecastResp.Predictions) {
			predicted = forecastResp.Predictions[index]
		}
		points = append(points, BacktestPoint{
			Timestamp: actual.Timestamp,
			Actual:    actual.Aggregate,
			Predicted: predicted,
		})
		actualValues = append(actualValues, actual.Aggregate)
		predictedValues = append(predictedValues, predicted)
	}

	return &BacktestResponse{
		ModelType:     request.ModelType,
		BacktestStart: request.BacktestStart,
		BacktestEnd:   request.BacktestEnd,
		Granularity:   request.Granularity,
		Predictions:   points,
		Metrics:       computeBacktestMetrics(actualValues, predictedValues),
	}, nil
}

func (c *HTTPClient) Health(ctx context.Context) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/internal/model/v1/health", nil)
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
		return fmt.Errorf("模型服务健康检查失败: status=%d body=%s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
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

func (c *HTTPClient) Backtest(ctx context.Context, request BacktestRequest) (*BacktestResponse, error) {
	var result BacktestResponse
	if err := c.doJSON(ctx, http.MethodPost, "/internal/model/v1/backtest", request, &result); err != nil {
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
		return fmt.Errorf("模型服务请求失败: status=%d body=%s", response.StatusCode, strings.TrimSpace(string(raw)))
	}
	return json.NewDecoder(response.Body).Decode(target)
}

func safeMean(total float64, count int) float64 {
	if count <= 0 {
		return 0
	}
	return total / float64(count)
}

func computeBacktestMetrics(actualValues, predictedValues []float64) BacktestMetrics {
	if len(actualValues) == 0 || len(actualValues) != len(predictedValues) {
		return BacktestMetrics{}
	}

	absErrorSum := 0.0
	squaredErrorSum := 0.0
	smapeSum := 0.0
	actualAbsSum := 0.0

	for index, actual := range actualValues {
		predicted := predictedValues[index]
		errValue := predicted - actual
		absError := math.Abs(errValue)
		absErrorSum += absError
		squaredErrorSum += errValue * errValue
		actualAbsSum += math.Abs(actual)

		denominator := math.Abs(actual) + math.Abs(predicted)
		if denominator > 1e-6 {
			smapeSum += (2 * absError) / denominator
		}
	}

	count := float64(len(actualValues))
	wape := 0.0
	if actualAbsSum > 1e-6 {
		wape = absErrorSum / actualAbsSum * 100
	}

	return BacktestMetrics{
		MAE:   absErrorSum / count,
		RMSE:  math.Sqrt(squaredErrorSum / count),
		SMAPE: smapeSum / count * 100,
		WAPE:  wape,
	}
}
