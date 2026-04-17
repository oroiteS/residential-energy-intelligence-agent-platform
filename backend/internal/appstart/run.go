package appstart

import (
	"context"
	"errors"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"residential-energy-intelligence-agent-platform/internal/bootstrap"
	"residential-energy-intelligence-agent-platform/internal/config"
)

// Run 启动后端 HTTP 服务，并负责优雅退出。
func Run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	app, cleanup, err := bootstrap.BuildApp(cfg)
	if err != nil {
		return err
	}
	defer cleanup()

	server := &http.Server{
		Addr:              cfg.ServerAddr,
		Handler:           app.Router,
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		app.Logger.Info("后端服务启动",
			bootstrap.FieldString("addr", cfg.ServerAddr),
			bootstrap.FieldString("app_name", cfg.AppName),
		)
		if serveErr := server.ListenAndServe(); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			app.Logger.Fatal("后端服务启动失败", bootstrap.FieldError(serveErr))
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	app.Logger.Info("后端服务关闭中")
	if err := server.Shutdown(ctx); err != nil {
		app.Logger.Error("后端服务关闭失败", bootstrap.FieldError(err))
		return err
	}
	return nil
}
