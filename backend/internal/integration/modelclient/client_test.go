package modelclient

import "testing"

func TestBuildCandidateBaseURLsFromLoopback(t *testing.T) {
	candidates := buildCandidateBaseURLs("http://127.0.0.1:8001")
	if len(candidates) < 2 {
		t.Fatalf("candidates = %#v, want multiple candidates", candidates)
	}
	if candidates[0] != "http://127.0.0.1:8001" {
		t.Fatalf("candidates[0] = %q, want loopback url", candidates[0])
	}
	if !containsURL(candidates, "http://host.docker.internal:8001") {
		t.Fatalf("candidates = %#v, want host.docker.internal fallback", candidates)
	}
	if !containsURL(candidates, "http://localhost:8001") {
		t.Fatalf("candidates = %#v, want localhost fallback", candidates)
	}
}

func TestBuildCandidateBaseURLsFromDockerHost(t *testing.T) {
	candidates := buildCandidateBaseURLs("http://host.docker.internal:8001")
	if len(candidates) < 2 {
		t.Fatalf("candidates = %#v, want multiple candidates", candidates)
	}
	if candidates[0] != "http://host.docker.internal:8001" {
		t.Fatalf("candidates[0] = %q, want host.docker.internal url", candidates[0])
	}
	if !containsURL(candidates, "http://127.0.0.1:8001") {
		t.Fatalf("candidates = %#v, want loopback fallback", candidates)
	}
}

func containsURL(items []string, target string) bool {
	for _, item := range items {
		if item == target {
			return true
		}
	}
	return false
}
