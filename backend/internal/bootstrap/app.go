package bootstrap

import (
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"residential-energy-intelligence-agent-platform/internal/config"
	"residential-energy-intelligence-agent-platform/internal/http/handler"
	"residential-energy-intelligence-agent-platform/internal/http/router"
	"residential-energy-intelligence-agent-platform/internal/integration/agentclient"
	"residential-energy-intelligence-agent-platform/internal/integration/modelclient"
	"residential-energy-intelligence-agent-platform/internal/job"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/internal/service"
)

type App struct {
	Router *gin.Engine
	Logger *zap.Logger
}

func BuildApp(cfg *config.AppConfig) (*App, func(), error) {
	logger, loggerCleanup, err := NewLogger(cfg.GinMode)
	if err != nil {
		return nil, nil, err
	}

	db, err := NewDatabase(cfg, logger)
	if err != nil {
		loggerCleanup()
		return nil, nil, err
	}

	executor := job.NewInMemoryExecutor()

	var systemConfigRepo repository.SystemConfigRepository
	var datasetRepo repository.DatasetRepository
	var analysisRepo repository.AnalysisResultRepository
	var classificationRepo repository.ClassificationResultRepository
	var forecastRepo repository.ForecastResultRepository
	var chatSessionRepo repository.ChatSessionRepository
	var chatMessageRepo repository.ChatMessageRepository
	var adviceRepo repository.EnergyAdviceRepository
	var reportRepo repository.ReportRepository
	if db != nil {
		systemConfigRepo = repository.NewSystemConfigRepository(db)
		datasetRepo = repository.NewDatasetRepository(db)
		analysisRepo = repository.NewAnalysisResultRepository(db)
		classificationRepo = repository.NewClassificationResultRepository(db)
		forecastRepo = repository.NewForecastResultRepository(db)
		chatSessionRepo = repository.NewChatSessionRepository(db)
		chatMessageRepo = repository.NewChatMessageRepository(db)
		adviceRepo = repository.NewEnergyAdviceRepository(db)
		reportRepo = repository.NewReportRepository(db)
	}

	var modelSvcClient modelclient.Client
	if cfg.UseStubClients || cfg.ModelServiceBaseURL == "" {
		modelSvcClient = modelclient.NewStubClient()
		logger.Info("模型客户端以 stub 模式启动")
	} else {
		modelSvcClient = modelclient.NewHTTPClient(cfg.ModelServiceBaseURL, cfg.RequestTimeout)
	}

	var agentSvcClient agentclient.Client
	if cfg.UseStubClients || cfg.AgentServiceBaseURL == "" {
		agentSvcClient = agentclient.NewStubClient()
		logger.Info("智能体客户端以 stub 模式启动")
	} else {
		agentTimeout := cfg.RequestTimeout
		if agentTimeout < 180*time.Second {
			agentTimeout = 180 * time.Second
		}
		agentSvcClient = agentclient.NewHTTPClient(cfg.AgentServiceBaseURL, agentTimeout)
	}

	healthService := service.NewHealthService(db, modelSvcClient, agentSvcClient)
	systemConfigService := service.NewSystemConfigService(db, systemConfigRepo, logger)
	chatService := service.NewChatService(cfg, datasetRepo, chatSessionRepo, chatMessageRepo)
	analysisService := service.NewAnalysisService(cfg, datasetRepo, analysisRepo, systemConfigService, logger)
	classificationService := service.NewClassificationService(cfg, datasetRepo, classificationRepo, systemConfigService, modelSvcClient, logger)
	forecastService := service.NewForecastService(cfg, datasetRepo, forecastRepo, systemConfigService, modelSvcClient, logger)
	agentService := service.NewAgentService(cfg, datasetRepo, chatSessionRepo, chatMessageRepo, analysisRepo, classificationRepo, forecastRepo, adviceRepo, systemConfigService, agentSvcClient, logger)
	adviceService := service.NewAdviceService(cfg, datasetRepo, analysisRepo, adviceRepo, logger)
	reportService := service.NewReportService(cfg, datasetRepo, analysisRepo, adviceRepo, reportRepo, agentService, logger)
	datasetService := service.NewDatasetService(cfg, datasetRepo, analysisRepo, forecastRepo, adviceRepo, reportRepo, executor, analysisService, logger)

	engine := router.New(router.Dependencies{
		Logger:                logger,
		HealthHandler:         handler.NewHealthHandler(healthService),
		SystemConfigHandler:   handler.NewSystemConfigHandler(systemConfigService),
		DatasetHandler:        handler.NewDatasetHandler(datasetService),
		ChatHandler:           handler.NewChatHandler(chatService),
		AnalysisHandler:       handler.NewAnalysisHandler(analysisService),
		ClassificationHandler: handler.NewClassificationHandler(classificationService),
		ForecastHandler:       handler.NewForecastHandler(forecastService),
		AgentHandler:          handler.NewAgentHandler(agentService),
		AdviceHandler:         handler.NewAdviceHandler(adviceService),
		ReportHandler:         handler.NewReportHandler(reportService),
	})

	cleanup := func() {
		executor.Shutdown()
		loggerCleanup()
	}

	return &App{
		Router: engine,
		Logger: logger,
	}, cleanup, nil
}
