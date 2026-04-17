package appstart

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"live/internal/liveapp"
)

// Run 启动 live 模块 HTTP 服务。
func Run() error {
	projectRoot, err := projectRoot()
	if err != nil {
		return err
	}

	port := getenv("LIVE_PORT", "8090")
	dataPath, err := resolveDataPath(projectRoot, getenv("LIVE_DATA_PATH", "data"))
	if err != nil {
		return fmt.Errorf("解析 live 数据文件失败: %w", err)
	}
	webPath := resolvePath(projectRoot, getenv("LIVE_WEB_PATH", "web"))
	modelServiceBaseURL := getenv("LIVE_MODEL_SERVICE_BASE_URL", "http://127.0.0.1:8001")
	agentServiceBaseURL := getenv("LIVE_AGENT_SERVICE_BASE_URL", modelServiceBaseURL)
	forecastModelType := getenv("LIVE_FORECAST_MODEL_TYPE", "tft")
	requestTimeout := getenvDurationSeconds("LIVE_REQUEST_TIMEOUT_SECONDS", 60)

	serviceClient := liveapp.NewServiceClient(
		modelServiceBaseURL,
		agentServiceBaseURL,
		forecastModelType,
		requestTimeout,
	)

	simulator, err := liveapp.NewSimulator(dataPath, time.Second, serviceClient)
	if err != nil {
		return fmt.Errorf("初始化 live 模拟器失败: %w", err)
	}
	simulator.Start()

	server := liveapp.NewServer(simulator, webPath)
	addr := ":" + port
	log.Printf("live 模块启动完成: http://127.0.0.1%s", addr)
	log.Printf("当前加载数据文件: %s", dataPath)

	if err := http.ListenAndServe(addr, server.Handler()); err != nil {
		return fmt.Errorf("启动 HTTP 服务失败: %w", err)
	}
	return nil
}

func getenv(key string, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func getenvDurationSeconds(key string, fallbackSeconds int) time.Duration {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return time.Duration(fallbackSeconds) * time.Second
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return time.Duration(fallbackSeconds) * time.Second
	}
	return time.Duration(parsed) * time.Second
}

func projectRoot() (string, error) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("无法解析 live 项目根目录")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(currentFile), "..", "..")), nil
}

func resolvePath(projectRoot string, value string) string {
	if filepath.IsAbs(value) {
		return value
	}
	return filepath.Join(projectRoot, value)
}

func resolveDataPath(projectRoot string, value string) (string, error) {
	resolved := resolvePath(projectRoot, value)
	info, err := os.Stat(resolved)
	if err == nil {
		if info.IsDir() {
			return pickLatestCSV(resolved)
		}
		if strings.EqualFold(filepath.Ext(resolved), ".csv") {
			return resolved, nil
		}
		return "", fmt.Errorf("LIVE_DATA_PATH 不是 csv 文件也不是目录: %s", resolved)
	}
	if !os.IsNotExist(err) {
		return "", err
	}
	return "", fmt.Errorf("未找到 LIVE_DATA_PATH 指向的文件或目录: %s", resolved)
}

func pickLatestCSV(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}

	type candidate struct {
		path    string
		modTime time.Time
	}

	candidates := make([]candidate, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.EqualFold(filepath.Ext(entry.Name()), ".csv") {
			continue
		}
		info, infoErr := entry.Info()
		if infoErr != nil {
			return "", infoErr
		}
		candidates = append(candidates, candidate{
			path:    filepath.Join(dir, entry.Name()),
			modTime: info.ModTime(),
		})
	}
	if len(candidates) == 0 {
		return "", fmt.Errorf("目录下没有可用 csv 文件: %s", dir)
	}

	sort.Slice(candidates, func(left, right int) bool {
		if candidates[left].modTime.Equal(candidates[right].modTime) {
			return candidates[left].path > candidates[right].path
		}
		return candidates[left].modTime.After(candidates[right].modTime)
	})
	return candidates[0].path, nil
}
