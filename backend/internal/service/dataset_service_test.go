package service

import "testing"

func TestResolveColumnMappingSupportsMultipleApplianceColumns(t *testing.T) {
	headers := []string{"Time", "Aggregate", "Appliance1", "Appliance2"}

	resolved, applianceColumns, appErr := resolveColumnMapping(headers, map[string]string{})
	if appErr != nil {
		t.Fatalf("resolveColumnMapping() 返回错误: %v", appErr)
	}

	if resolved["Time"] != "timestamp" {
		t.Fatalf("Time semantic = %s, want timestamp", resolved["Time"])
	}
	if resolved["Aggregate"] != "aggregate" {
		t.Fatalf("Aggregate semantic = %s, want aggregate", resolved["Aggregate"])
	}
	if len(applianceColumns) != 2 {
		t.Fatalf("applianceColumns len = %d, want 2", len(applianceColumns))
	}
}

func TestResolveColumnMappingRequiresTimestampAndAggregate(t *testing.T) {
	headers := []string{"foo", "bar"}

	_, _, appErr := resolveColumnMapping(headers, map[string]string{})
	if appErr == nil {
		t.Fatal("resolveColumnMapping() 预期返回错误，实际为 nil")
	}
	if appErr.Code != "COLUMN_MAPPING_REQUIRED" {
		t.Fatalf("appErr.Code = %s, want COLUMN_MAPPING_REQUIRED", appErr.Code)
	}
}
