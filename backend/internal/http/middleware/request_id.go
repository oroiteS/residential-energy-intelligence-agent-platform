package middleware

import (
	"fmt"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
)

var requestCounter atomic.Uint64

func RequestID() gin.HandlerFunc {
	return func(c *gin.Context) {
		id := fmt.Sprintf("req_%s_%06d", time.Now().Format("20060102_150405"), requestCounter.Add(1))
		c.Set("request_id", id)
		c.Writer.Header().Set("X-Request-ID", id)
		c.Next()
	}
}
