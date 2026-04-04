package service

import "testing"

func TestUniqueStringsKeepsOrderAndRemovesDuplicates(t *testing.T) {
	result := uniqueStrings([]string{"a", "b", "a", "c", "b"})
	if len(result) != 3 {
		t.Fatalf("len(result) = %d, want 3", len(result))
	}
	if result[0] != "a" || result[1] != "b" || result[2] != "c" {
		t.Fatalf("unexpected result: %#v", result)
	}
}
