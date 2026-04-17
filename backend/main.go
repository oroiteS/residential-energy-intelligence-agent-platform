package main

import (
	"log"

	"residential-energy-intelligence-agent-platform/internal/appstart"
)

func main() {
	if err := appstart.Run(); err != nil {
		log.Fatal(err)
	}
}
