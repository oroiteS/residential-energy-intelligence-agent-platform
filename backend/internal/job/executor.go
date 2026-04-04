package job

import (
	"context"
	"sync"
)

type Executor interface {
	Submit(ctx context.Context, fn func(context.Context))
	Shutdown()
}

type InMemoryExecutor struct {
	wg sync.WaitGroup
}

func NewInMemoryExecutor() *InMemoryExecutor {
	return &InMemoryExecutor{}
}

func (e *InMemoryExecutor) Submit(ctx context.Context, fn func(context.Context)) {
	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		fn(ctx)
	}()
}

func (e *InMemoryExecutor) Shutdown() {
	e.wg.Wait()
}
