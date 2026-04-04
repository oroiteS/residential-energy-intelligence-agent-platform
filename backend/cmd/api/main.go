package main

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

func main() {
	cfg, err := config.Load()
	if err != nil {
		panic(err)
	}

	app, cleanup, err := bootstrap.BuildApp(cfg)
	if err != nil {
		panic(err)
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
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			app.Logger.Fatal("后端服务启动失败", bootstrap.FieldError(err))
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
	}
}
