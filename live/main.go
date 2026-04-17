package main

import (
	"log"

	"live/internal/appstart"
)

func main() {
	if err := appstart.Run(); err != nil {
		log.Fatal(err)
	}
}
