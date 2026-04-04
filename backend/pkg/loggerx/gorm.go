package loggerx

import (
	"context"
	"time"

	"go.uber.org/zap"
	"gorm.io/gorm/logger"
)

type GormZapLogger struct {
	logger        *zap.Logger
	slowThreshold time.Duration
	logLevel      logger.LogLevel
}

func NewGormZapLogger(base *zap.Logger) logger.Interface {
	return &GormZapLogger{
		logger:        base.Named("gorm"),
		slowThreshold: 500 * time.Millisecond,
		logLevel:      logger.Warn,
	}
}

func (l *GormZapLogger) LogMode(level logger.LogLevel) logger.Interface {
	cloned := *l
	cloned.logLevel = level
	return &cloned
}

func (l *GormZapLogger) Info(_ context.Context, msg string, args ...interface{}) {
	if l.logLevel < logger.Info {
		return
	}
	l.logger.Sugar().Infof(msg, args...)
}

func (l *GormZapLogger) Warn(_ context.Context, msg string, args ...interface{}) {
	if l.logLevel < logger.Warn {
		return
	}
	l.logger.Sugar().Warnf(msg, args...)
}

func (l *GormZapLogger) Error(_ context.Context, msg string, args ...interface{}) {
	if l.logLevel < logger.Error {
		return
	}
	l.logger.Sugar().Errorf(msg, args...)
}

func (l *GormZapLogger) Trace(_ context.Context, begin time.Time, fc func() (string, int64), err error) {
	if l.logLevel == logger.Silent {
		return
	}

	elapsed := time.Since(begin)
	sql, rows := fc()
	fields := []zap.Field{
		zap.Duration("elapsed", elapsed),
		zap.Int64("rows", rows),
		zap.String("sql", sql),
	}

	if err != nil {
		l.logger.Error("SQL 执行失败", append(fields, zap.Error(err))...)
		return
	}

	if elapsed > l.slowThreshold {
		l.logger.Warn("慢 SQL", fields...)
		return
	}

	if l.logLevel >= logger.Info {
		l.logger.Debug("SQL 执行完成", fields...)
	}
}
