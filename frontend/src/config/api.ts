/**
 * API configuration constants.
 *
 * Centralized configuration for API endpoints to avoid hardcoded URLs.
 */

/**
 * Default API base URL for the backend server.
 * Used by all API hooks to construct endpoint URLs.
 */
export const API_BASE = "http://127.0.0.1:8000";

/**
 * Test API base URL - same as API_BASE but exported separately
 * for use in test files to make it clear it's for testing.
 */
export const TEST_API_BASE = API_BASE;
