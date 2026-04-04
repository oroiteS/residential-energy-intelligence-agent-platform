package router

import (
	ginzap "github.com/gin-contrib/zap"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"residential-energy-intelligence-agent-platform/internal/http/handler"
	"residential-energy-intelligence-agent-platform/internal/http/middleware"
)

type Dependencies struct {
	Logger                *zap.Logger
	HealthHandler         *handler.HealthHandler
	SystemConfigHandler   *handler.SystemConfigHandler
	LLMConfigHandler      *handler.LLMConfigHandler
	DatasetHandler        *handler.DatasetHandler
	ChatHandler           *handler.ChatHandler
	AnalysisHandler       *handler.AnalysisHandler
	ClassificationHandler *handler.ClassificationHandler
	ForecastHandler       *handler.ForecastHandler
	AgentHandler          *handler.AgentHandler
	AdviceHandler         *handler.AdviceHandler
	ReportHandler         *handler.ReportHandler
}

func New(dep Dependencies) *gin.Engine {
	engine := gin.New()
	engine.Use(middleware.RequestID())
	engine.Use(ginzap.Ginzap(dep.Logger, "", true))
	engine.Use(middleware.Recovery(dep.Logger))

	api := engine.Group("/api/v1")
	{
		api.GET("/health", dep.HealthHandler.Get)
		api.GET("/system/config", dep.SystemConfigHandler.Get)
		api.PATCH("/system/config", dep.SystemConfigHandler.Patch)
		api.GET("/llm-configs", dep.LLMConfigHandler.List)
		api.POST("/llm-configs", dep.LLMConfigHandler.Create)
		api.PUT("/llm-configs/:id", dep.LLMConfigHandler.Update)
		api.DELETE("/llm-configs/:id", dep.LLMConfigHandler.Delete)
		api.POST("/llm-configs/:id/set-default", dep.LLMConfigHandler.SetDefault)
		api.POST("/datasets/import", dep.DatasetHandler.Import)
		api.POST("/chat/sessions", dep.ChatHandler.CreateSession)
		api.GET("/chat/sessions", dep.ChatHandler.ListSessions)
		api.DELETE("/chat/sessions/:id", dep.ChatHandler.DeleteSession)
		api.GET("/chat/sessions/:id/messages", dep.ChatHandler.ListMessages)
		api.GET("/datasets", dep.DatasetHandler.List)
		api.GET("/datasets/:id", dep.DatasetHandler.Get)
		api.DELETE("/datasets/:id", dep.DatasetHandler.Delete)
		api.GET("/datasets/:id/analysis", dep.AnalysisHandler.Get)
		api.POST("/datasets/:id/analysis/recompute", dep.AnalysisHandler.Recompute)
		api.POST("/datasets/:id/classifications/predict", dep.ClassificationHandler.Predict)
		api.GET("/datasets/:id/classifications/latest", dep.ClassificationHandler.GetLatest)
		api.GET("/datasets/:id/classifications", dep.ClassificationHandler.List)
		api.POST("/datasets/:id/forecasts/predict", dep.ForecastHandler.Predict)
		api.GET("/datasets/:id/forecasts", dep.ForecastHandler.List)
		api.GET("/forecasts/:forecast_id", dep.ForecastHandler.Get)
		api.POST("/datasets/:id/forecasts/backtest", dep.ForecastHandler.Backtest)
		api.POST("/agent/ask", dep.AgentHandler.Ask)
		api.GET("/datasets/:id/advices", dep.AdviceHandler.List)
		api.POST("/datasets/:id/advices/generate", dep.AdviceHandler.Generate)
		api.GET("/advices/:id", dep.AdviceHandler.Get)
		api.POST("/datasets/:id/reports/export", dep.ReportHandler.Export)
		api.GET("/datasets/:id/reports", dep.ReportHandler.List)
		api.GET("/reports/:id/download", dep.ReportHandler.Download)
	}

	engine.GET("/", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"service": "resident-energy-backend",
			"status":  "up",
		})
	})
	return engine
}
