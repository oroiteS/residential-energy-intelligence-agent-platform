package bootstrap

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func NewLogger(mode string) (*zap.Logger, func(), error) {
	var cfg zap.Config
	if mode == "release" {
		cfg = zap.NewProductionConfig()
		cfg.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	} else {
		cfg = zap.NewDevelopmentConfig()
		cfg.Level = zap.NewAtomicLevelAt(zap.DebugLevel)
	}

	cfg.Encoding = "json"
	cfg.OutputPaths = []string{"stdout"}
	cfg.ErrorOutputPaths = []string{"stderr"}
	cfg.EncoderConfig.TimeKey = "time"
	cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

	logger, err := cfg.Build(zap.AddCaller(), zap.AddStacktrace(zap.ErrorLevel))
	if err != nil {
		return nil, nil, err
	}

	cleanup := func() {
		_ = logger.Sync()
	}

	return logger, cleanup, nil
}

func FieldError(err error) zap.Field {
	return zap.Error(err)
}

func FieldString(key, value string) zap.Field {
	return zap.String(key, value)
}
