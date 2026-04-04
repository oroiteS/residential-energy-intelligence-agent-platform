package service

import (
	"context"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/integration/agentclient"
	"residential-energy-intelligence-agent-platform/internal/integration/modelclient"
)

type HealthService struct {
	db          *gorm.DB
	modelClient modelclient.Client
	agentClient agentclient.Client
}

type HealthStatus struct {
	Service      string            `json:"service"`
	Status       string            `json:"status"`
	Version      string            `json:"version"`
	Dependencies map[string]string `json:"dependencies,omitempty"`
}

func NewHealthService(db *gorm.DB, modelClient modelclient.Client, agentClient agentclient.Client) *HealthService {
	return &HealthService{
		db:          db,
		modelClient: modelClient,
		agentClient: agentClient,
	}
}

func (s *HealthService) GetStatus(ctx context.Context) HealthStatus {
	status := "up"
	dependencies := map[string]string{
		"database": "up",
		"model":    "up",
		"agent":    "up",
	}

	if s.db == nil {
		status = "degraded"
		dependencies["database"] = "degraded"
	}
	if s.modelClient != nil {
		if err := s.modelClient.Health(ctx); err != nil {
			status = "degraded"
			dependencies["model"] = "degraded"
		}
	}
	if s.agentClient != nil {
		if err := s.agentClient.Health(ctx); err != nil {
			status = "degraded"
			dependencies["agent"] = "degraded"
		}
	}

	return HealthStatus{
		Service:      "go-api",
		Status:       status,
		Version:      "v1",
		Dependencies: dependencies,
	}
}
