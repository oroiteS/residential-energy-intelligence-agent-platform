package liveapp

import (
	"fmt"
	"testing"
)

func TestTickCarriesPreviousNextDayForecastIntoTodayForecast(t *testing.T) {
	days := buildSequentialTestDays(forecastHistoryDays * 2)
	currentDayIndex := forecastHistoryDays

	expectedPoints := []ForecastPoint{
		{SlotIndex: 0, TimeLabel: "00:00", Predicted: 123.45},
		{SlotIndex: 1, TimeLabel: "00:15", Predicted: 234.56},
	}
	simulator := &Simulator{
		days:        days,
		dayIndex:    currentDayIndex,
		slotIndex:   daySlots - 1,
		running:     true,
		subscribers: make(map[chan Snapshot]struct{}),
		todayForecast: DayForecast{
			Date:      days[currentDayIndex].Date,
			ModelType: "tft",
		},
		nextDayForecast: DayForecast{
			Date:            days[currentDayIndex+1].Date,
			Points:          expectedPoints,
			AvgLoadW:        345.67,
			PeakLoadW:       456.78,
			PeakPeriods:     []string{"18:00"},
			RiskFlags:       []string{"evening_peak"},
			Explanation:     "上一日生成的次日预测",
			HistoryDaysUsed: forecastHistoryDays,
			ModelType:       "tft",
		},
	}

	snapshot := simulator.Tick()

	if snapshot.VirtualDate != days[currentDayIndex+1].Date {
		t.Fatalf("VirtualDate = %s, want %s", snapshot.VirtualDate, days[currentDayIndex+1].Date)
	}
	if snapshot.TodayForecast.Date != days[currentDayIndex+1].Date {
		t.Fatalf("TodayForecast.Date = %s, want %s", snapshot.TodayForecast.Date, days[currentDayIndex+1].Date)
	}
	if snapshot.TodayForecast.AvgLoadW != 345.67 {
		t.Fatalf("TodayForecast.AvgLoadW = %v, want 345.67", snapshot.TodayForecast.AvgLoadW)
	}
	if len(snapshot.TodayForecast.Points) != len(expectedPoints) {
		t.Fatalf("len(TodayForecast.Points) = %d, want %d", len(snapshot.TodayForecast.Points), len(expectedPoints))
	}
	for index, point := range expectedPoints {
		if snapshot.TodayForecast.Points[index] != point {
			t.Fatalf("TodayForecast.Points[%d] = %#v, want %#v", index, snapshot.TodayForecast.Points[index], point)
		}
	}
}

func buildSequentialTestDays(totalDays int) []daySeries {
	days := make([]daySeries, 0, totalDays)
	for index := 0; index < totalDays; index++ {
		days = append(days, buildTestDaySeries(fmt.Sprintf("2024-01-%02d", index+1)))
	}
	return days
}

func buildTestDaySeries(date string) daySeries {
	points := make([]DataPoint, 0, daySlots)
	for index := 0; index < daySlots; index++ {
		points = append(points, DataPoint{
			Date:      date,
			SlotIndex: index,
			TimeLabel: slotLabel(index),
			Timestamp: date + " " + slotLabel(index) + ":00",
			Aggregate: float64(index),
		})
	}
	return daySeries{
		Date:   date,
		Points: points,
	}
}

func slotLabel(index int) string {
	hour := index / 4
	minute := (index % 4) * 15
	return twoDigit(hour) + ":" + twoDigit(minute)
}

func twoDigit(value int) string {
	if value < 10 {
		return "0" + string(rune('0'+value))
	}
	return string(rune('0'+value/10)) + string(rune('0'+value%10))
}
