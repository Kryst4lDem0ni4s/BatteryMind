#!/bin/bash

# BatteryMind Health Check Script
# Comprehensive system health monitoring for deployed models
#
# Author: BatteryMind Development Team
# Version: 1.0.0
# Dependencies: AWS CLI, curl, jq

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_FILE="/var/log/batterymind/health_check.log"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_TIMEOUT=120
DEFAULT_ENVIRONMENT="staging"

# AWS Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_MODEL_BUCKET="${S3_MODEL_BUCKET:-batterymind-models}"
SAGEMAKER_ENDPOINT_PREFIX="${SAGEMAKER_ENDPOINT_PREFIX:-batterymind}"

# Health check results
declare -A HEALTH_RESULTS
OVERALL_HEALTH="healthy"
FAILED_CHECKS=0
WARNING_CHECKS=0
TOTAL_CHECKS=0

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Setup logging
setup_logging() {
    local log_dir
    log_dir="$(dirname "$LOG_FILE")"
    
    if [[ ! -d "$log_dir" ]]; then
        sudo mkdir -p "$log_dir" || {
            LOG_FILE="./health_check_${TIMESTAMP}.log"
            log_warn "Cannot create system log directory, using local log file: $LOG_FILE"
        }
    fi
    
    log_info "Health check started - Session ID: $TIMESTAMP"
}

# Display usage information
usage() {
    cat << EOF
BatteryMind Health Check Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE           Model type to check (transformer, federated, rl, ensemble, all)
    -e, --env ENVIRONMENT     Target environment (dev, staging, prod)
    -c, --checks CHECKS       Comma-separated list of checks to run
    -T, --timeout SECONDS     Timeout for individual checks (default: 120)
    -v, --verbose            Enable verbose output
    -j, --json               Output results in JSON format
    -w, --warnings-as-errors  Treat warnings as errors
    -h, --help               Show this help message

AVAILABLE CHECKS:
    aws                      AWS connectivity and permissions
    sagemaker               SageMaker endpoint health
    s3                      S3 bucket accessibility
    model                   Model artifact integrity
    inference               Inference functionality
    performance             Performance benchmarks
    monitoring              Monitoring systems
    security                Security configurations

EXAMPLES:
    $0 -t transformer -e prod
    $0 --type all --env staging --checks aws,sagemaker,inference
    $0 -t rl -e prod --timeout 60 --json

EOF
}

# Parse command line arguments
parse_arguments() {
    PARSED_ARGS=$(getopt -o t:e:c:T:vjwh --long type:,env:,checks:,timeout:,verbose,json,warnings-as-errors,help -n "$0" -- "$@")
    
    if [[ $? -ne 0 ]]; then
        echo "Invalid arguments provided" >&2
        exit 1
    fi
    
    eval set -- "$PARSED_ARGS"
    
    while true; do
        case "$1" in
            -t|--type)
                MODEL_TYPE="$2"
                shift 2
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -c|--checks)
                SPECIFIC_CHECKS="$2"
                shift 2
                ;;
            -T|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -j|--json)
                JSON_OUTPUT=true
                shift
                ;;
            -w|--warnings-as-errors)
                WARNINGS_AS_ERRORS=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Unknown option: $1" >&2
                exit 1
                ;;
        esac
    done
    
    # Set defaults
    MODEL_TYPE="${MODEL_TYPE:-all}"
    ENVIRONMENT="${ENVIRONMENT:-$DEFAULT_ENVIRONMENT}"
    TIMEOUT="${TIMEOUT:-$DEFAULT_TIMEOUT}"
    VERBOSE="${VERBOSE:-false}"
    JSON_OUTPUT="${JSON_OUTPUT:-false}"
    WARNINGS_AS_ERRORS="${WARNINGS_AS_ERRORS:-false}"
}

# Record check result
record_result() {
    local check_name="$1"
    local status="$2"
    local message="$3"
    local details="${4:-}"
    
    HEALTH_RESULTS["$check_name"]="$status|$message|$details"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    case "$status" in
        "pass")
            log_pass "$check_name: $message"
            ;;
        "warn")
            log_warn "$check_name: $message"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
            if [[ "$WARNINGS_AS_ERRORS" == "true" ]]; then
                OVERALL_HEALTH="unhealthy"
                FAILED_CHECKS=$((FAILED_CHECKS + 1))
            fi
            ;;
        "fail")
            log_fail "$check_name: $message"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            OVERALL_HEALTH="unhealthy"
            ;;
    esac
}

# Check AWS connectivity and permissions
check_aws_connectivity() {
    log_info "Checking AWS connectivity and permissions..."
    
    # Check AWS CLI configuration
    if ! command -v aws &> /dev/null; then
        record_result "aws_cli" "fail" "AWS CLI not installed"
        return 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        record_result "aws_credentials" "fail" "AWS credentials not configured or invalid"
        return 1
    fi
    
    record_result "aws_credentials" "pass" "AWS credentials valid"
    
    # Check S3 bucket access
    if aws s3 ls "s3://$S3_MODEL_BUCKET" &> /dev/null; then
        record_result "s3_bucket_access" "pass" "S3 bucket accessible"
    else
        record_result "s3_bucket_access" "fail" "Cannot access S3 bucket: $S3_MODEL_BUCKET"
        return 1
    fi
    
    # Check SageMaker permissions
    if aws sagemaker list-endpoints --region "$AWS_REGION" &> /dev/null; then
        record_result "sagemaker_permissions" "pass" "SageMaker permissions valid"
    else
        record_result "sagemaker_permissions" "fail" "Insufficient SageMaker permissions"
        return 1
    fi
    
    log_info "AWS connectivity checks completed"
    return 0
}

# Check SageMaker endpoint health
check_sagemaker_health() {
    log_info "Checking SageMaker endpoint health..."
    
    local model_types=()
    if [[ "$MODEL_TYPE" == "all" ]]; then
        model_types=("transformer" "federated" "rl" "ensemble")
    else
        model_types=("$MODEL_TYPE")
    fi
    
    for type in "${model_types[@]}"; do
        local endpoint_name="${SAGEMAKER_ENDPOINT_PREFIX}-${type}-${ENVIRONMENT}"
        
        # Check if endpoint exists
        if ! aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" --region "$AWS_REGION" &> /dev/null; then
            record_result "sagemaker_${type}_exists" "warn" "Endpoint does not exist: $endpoint_name"
            continue
        fi
        
        # Check endpoint status
        local status
        status=$(aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" --region "$AWS_REGION" --query 'EndpointStatus' --output text)
        
        case "$status" in
            "InService")
                record_result "sagemaker_${type}_status" "pass" "Endpoint is in service: $endpoint_name"
                ;;
            "Creating"|"Updating")
                record_result "sagemaker_${type}_status" "warn" "Endpoint is updating: $endpoint_name ($status)"
                ;;
            "Failed"|"OutOfService")
                record_result "sagemaker_${type}_status" "fail" "Endpoint is unhealthy: $endpoint_name ($status)"
                ;;
            *)
                record_result "sagemaker_${type}_status" "warn" "Endpoint status unknown: $endpoint_name ($status)"
                ;;
        esac
        
        # Check endpoint configuration
        if [[ "$status" == "InService" ]]; then
            local instance_count
            instance_count=$(aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" --region "$AWS_REGION" --query 'ProductionVariants[0].CurrentInstanceCount' --output text)
            
            if [[ "$instance_count" -gt 0 ]]; then
                record_result "sagemaker_${type}_instances" "pass" "Endpoint has $instance_count active instances"
            else
                record_result "sagemaker_${type}_instances" "fail" "Endpoint has no active instances"
            fi
        fi
    done
    
    log_info "SageMaker health checks completed"
}

# Check S3 model artifacts
check_s3_model_artifacts() {
    log_info "Checking S3 model artifacts..."
    
    local model_types=()
    if [[ "$MODEL_TYPE" == "all" ]]; then
        model_types=("transformer" "federated" "rl" "ensemble")
    else
        model_types=("$MODEL_TYPE")
    fi
    
    for type in "${model_types[@]}"; do
        local current_path="s3://$S3_MODEL_BUCKET/models/${type}/current/"
        
        # Check if current model exists
        if aws s3 ls "$current_path" &> /dev/null; then
            record_result "s3_${type}_current" "pass" "Current model artifacts exist: $current_path"
            
            # Check for required files
            local required_files
            case "$type" in
                "transformer")
                    required_files=("model.pkl" "config.json" "model_metadata.yaml")
                    ;;
                "federated")
                    required_files=("global_model.pkl" "client_configs.json" "privacy_params.yaml")
                    ;;
                "rl")
                    required_files=("policy_network.pt" "value_network.pt" "environment_config.yaml")
                    ;;
                "ensemble")
                    required_files=("ensemble_model.pkl" "ensemble_config.json")
                    ;;
            esac
            
            local missing_files=()
            for file in "${required_files[@]}"; do
                if ! aws s3 ls "${current_path}${file}" &> /dev/null; then
                    missing_files+=("$file")
                fi
            done
            
            if [[ ${#missing_files[@]} -eq 0 ]]; then
                record_result "s3_${type}_integrity" "pass" "All required model files present"
            else
                record_result "s3_${type}_integrity" "fail" "Missing required files: ${missing_files[*]}"
            fi
            
            # Check model version
            local version
            version=$(aws s3 cp "${current_path}version.txt" - 2>/dev/null || echo "unknown")
            if [[ "$version" != "unknown" ]]; then
                record_result "s3_${type}_version" "pass" "Model version: $version"
            else
                record_result "s3_${type}_version" "warn" "Model version information missing"
            fi
        else
            record_result "s3_${type}_current" "fail" "Current model artifacts not found: $current_path"
        fi
    done
    
    log_info "S3 model artifact checks completed"
}

# Check model inference functionality
check_model_inference() {
    log_info "Checking model inference functionality..."
    
    local model_types=()
    if [[ "$MODEL_TYPE" == "all" ]]; then
        model_types=("transformer" "federated" "rl" "ensemble")
    else
        model_types=("$MODEL_TYPE")
    fi
    
    for type in "${model_types[@]}"; do
        local endpoint_name="${SAGEMAKER_ENDPOINT_PREFIX}-${type}-${ENVIRONMENT}"
        
        # Skip if endpoint doesn't exist or isn't in service
        local endpoint_status
        endpoint_status=$(aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" --region "$AWS_REGION" --query 'EndpointStatus' --output text 2>/dev/null || echo "NotFound")
        
        if [[ "$endpoint_status" != "InService" ]]; then
            record_result "inference_${type}_availability" "warn" "Endpoint not available for inference testing"
            continue
        fi
        
        # Create test payload based on model type
        local test_payload
        case "$type" in
            "transformer")
                test_payload='{"instances": [{"voltage": 3.7, "current": 2.5, "temperature": 25.0, "soc": 0.8}]}'
                ;;
            "federated")
                test_payload='{"client_id": "test_client", "local_updates": [0.1, 0.2, 0.3]}'
                ;;
            "rl")
                test_payload='{"state": [0.8, 25.0, 3.7, 2.5], "action_space": "charging"}'
                ;;
            "ensemble")
                test_payload='{"models": ["transformer", "rl"], "input": {"voltage": 3.7, "current": 2.5}}'
                ;;
        esac
        
        # Test inference with timeout
        local inference_start
        inference_start=$(date +%s)
        
        if timeout "$TIMEOUT" aws sagemaker-runtime invoke-endpoint \
            --endpoint-name "$endpoint_name" \
            --content-type "application/json" \
            --body "$test_payload" \
            --region "$AWS_REGION" \
            "/tmp/inference_result_${type}_${TIMESTAMP}.json" &> /dev/null; then
            
            local inference_end
            inference_end=$(date +%s)
            local inference_time
            inference_time=$((inference_end - inference_start))
            
            # Check response
            if [[ -f "/tmp/inference_result_${type}_${TIMESTAMP}.json" ]]; then
                local response_size
                response_size=$(wc -c < "/tmp/inference_result_${type}_${TIMESTAMP}.json")
                
                if [[ $response_size -gt 0 ]]; then
                    record_result "inference_${type}_functionality" "pass" "Inference successful (${inference_time}s, ${response_size} bytes)"
                    
                    # Check response time
                    if [[ $inference_time -le 5 ]]; then
                        record_result "inference_${type}_performance" "pass" "Response time within limits: ${inference_time}s"
                    elif [[ $inference_time -le 10 ]]; then
                        record_result "inference_${type}_performance" "warn" "Response time acceptable: ${inference_time}s"
                    else
                        record_result "inference_${type}_performance" "fail" "Response time too slow: ${inference_time}s"
                    fi
                else
                    record_result "inference_${type}_functionality" "fail" "Empty response from inference"
                fi
                
                # Cleanup
                rm -f "/tmp/inference_result_${type}_${TIMESTAMP}.json"
            else
                record_result "inference_${type}_functionality" "fail" "No response file generated"
            fi
        else
            record_result "inference_${type}_functionality" "fail" "Inference request failed or timed out"
        fi
    done
    
    log_info "Model inference checks completed"
}

# Check performance benchmarks
check_performance_benchmarks() {
    log_info "Checking performance benchmarks..."
    
    local model_types=()
    if [[ "$MODEL_TYPE" == "all" ]]; then
        model_types=("transformer" "federated" "rl" "ensemble")
    else
        model_types=("$MODEL_TYPE")
    fi
    
    for type in "${model_types[@]}"; do
        local endpoint_name="${SAGEMAKER_ENDPOINT_PREFIX}-${type}-${ENVIRONMENT}"
        
        # Skip if endpoint not available
        local endpoint_status
        endpoint_status=$(aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" --region "$AWS_REGION" --query 'EndpointStatus' --output text 2>/dev/null || echo "NotFound")
        
        if [[ "$endpoint_status" != "InService" ]]; then
            record_result "performance_${type}_availability" "warn" "Endpoint not available for performance testing"
            continue
        fi
        
        # Run multiple inference requests to test performance
        local total_time=0
        local successful_requests=0
        local failed_requests=0
        local test_iterations=5
        
        for ((i=1; i<=test_iterations; i++)); do
            local test_payload='{"instances": [{"test": "performance"}]}'
            local start_time
            start_time=$(date +%s%N)
            
            if timeout 10 aws sagemaker-runtime invoke-endpoint \
                --endpoint-name "$endpoint_name" \
                --content-type "application/json" \
                --body "$test_payload" \
                --region "$AWS_REGION" \
                "/tmp/perf_test_${type}_${i}_${TIMESTAMP}.json" &> /dev/null; then
                
                local end_time
                end_time=$(date +%s%N)
                local request_time
                request_time=$(( (end_time - start_time) / 1000000 ))  # Convert to milliseconds
                
                total_time=$((total_time + request_time))
                successful_requests=$((successful_requests + 1))
                
                # Cleanup
                rm -f "/tmp/perf_test_${type}_${i}_${TIMESTAMP}.json"
            else
                failed_requests=$((failed_requests + 1))
            fi
        done
        
        if [[ $successful_requests -gt 0 ]]; then
            local avg_time
            avg_time=$((total_time / successful_requests))
            local success_rate
            success_rate=$(( (successful_requests * 100) / test_iterations ))
            
            # Evaluate performance
            if [[ $avg_time -le 1000 && $success_rate -ge 95 ]]; then
                record_result "performance_${type}_benchmark" "pass" "Avg response: ${avg_time}ms, Success rate: ${success_rate}%"
            elif [[ $avg_time -le 3000 && $success_rate -ge 90 ]]; then
                record_result "performance_${type}_benchmark" "warn" "Avg response: ${avg_time}ms, Success rate: ${success_rate}%"
            else
                record_result "performance_${type}_benchmark" "fail" "Avg response: ${avg_time}ms, Success rate: ${success_rate}%"
            fi
        else
            record_result "performance_${type}_benchmark" "fail" "All performance test requests failed"
        fi
    done
    
    log_info "Performance benchmark checks completed"
}

# Check monitoring systems
check_monitoring_systems() {
    log_info "Checking monitoring systems..."
    
    # Check CloudWatch metrics
    local model_types=()
    if [[ "$MODEL_TYPE" == "all" ]]; then
        model_types=("transformer" "federated" "rl" "ensemble")
    else
        model_types=("$MODEL_TYPE")
    fi
    
    for type in "${model_types[@]}"; do
        local endpoint_name="${SAGEMAKER_ENDPOINT_PREFIX}-${type}-${ENVIRONMENT}"
        
        # Check if CloudWatch metrics are available
        local metrics_exist
        metrics_exist=$(aws cloudwatch list-metrics \
            --namespace "AWS/SageMaker" \
            --dimensions Name=EndpointName,Value="$endpoint_name" \
            --region "$AWS_REGION" \
            --query 'Metrics[0].MetricName' \
            --output text 2>/dev/null || echo "None")
        
        if [[ "$metrics_exist" != "None" ]]; then
            record_result "monitoring_${type}_cloudwatch" "pass" "CloudWatch metrics available for $endpoint_name"
        else
            record_result "monitoring_${type}_cloudwatch" "warn" "No CloudWatch metrics found for $endpoint_name"
        fi
    done
    
    # Check if custom monitoring scripts are running
    if pgrep -f "batterymind.*monitor" > /dev/null; then
        record_result "monitoring_custom_processes" "pass" "Custom monitoring processes are running"
    else
        record_result "monitoring_custom_processes" "warn" "No custom monitoring processes detected"
    fi
    
    log_info "Monitoring system checks completed"
}

# Check security configurations
check_security_configurations() {
    log_info "Checking security configurations..."
    
    # Check S3 bucket encryption
    local encryption_status
    encryption_status=$(aws s3api get-bucket-encryption --bucket "$S3_MODEL_BUCKET" --region "$AWS_REGION" 2>/dev/null || echo "NotEncrypted")
    
    if [[ "$encryption_status" != "NotEncrypted" ]]; then
        record_result "security_s3_encryption" "pass" "S3 bucket encryption enabled"
    else
        record_result "security_s3_encryption" "warn" "S3 bucket encryption not configured"
    fi
    
    # Check VPC configuration for SageMaker endpoints
    local model_types=()
    if [[ "$MODEL_TYPE" == "all" ]]; then
        model_types=("transformer" "federated" "rl" "ensemble")
    else
        model_types=("$MODEL_TYPE")
    fi
    
    for type in "${model_types[@]}"; do
        local endpoint_name="${SAGEMAKER_ENDPOINT_PREFIX}-${type}-${ENVIRONMENT}"
        
        # Check if endpoint exists and get VPC config
        if aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" --region "$AWS_REGION" &> /dev/null; then
            local vpc_config
            vpc_config=$(aws sagemaker describe-endpoint-config \
                --endpoint-config-name "${endpoint_name}-config" \
                --region "$AWS_REGION" \
                --query 'ProductionVariants[0].VpcConfig' \
                --output text 2>/dev/null || echo "None")
            
            if [[ "$vpc_config" != "None" ]]; then
                record_result "security_${type}_vpc" "pass" "VPC configuration enabled for $endpoint_name"
            else
                record_result "security_${type}_vpc" "warn" "No VPC configuration for $endpoint_name"
            fi
        fi
    done
    
    log_info "Security configuration checks completed"
}

# Generate health check report
generate_health_report() {
    local report_file="/var/log/batterymind/health_report_${TIMESTAMP}.json"
    
    # Prepare results for JSON output
    local results_json="{"
    local first=true
    
    for check in "${!HEALTH_RESULTS[@]}"; do
        if [[ "$first" == "true" ]]; then
            first=false
        else
            results_json+=","
        fi
        
        local result="${HEALTH_RESULTS[$check]}"
        local status="${result%%|*}"
        local temp="${result#*|}"
        local message="${temp%%|*}"
        local details="${temp#*|}"
        
        results_json+="\"$check\":{\"status\":\"$status\",\"message\":\"$message\",\"details\":\"$details\"}"
    done
    results_json+="}"
    
    # Generate comprehensive report
    cat > "$report_file" << EOF
{
    "health_check_session_id": "$TIMESTAMP",
    "model_type": "$MODEL_TYPE",
    "environment": "$ENVIRONMENT",
    "check_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "overall_health": "$OVERALL_HEALTH",
    "summary": {
        "total_checks": $TOTAL_CHECKS,
        "passed_checks": $((TOTAL_CHECKS - FAILED_CHECKS - WARNING_CHECKS)),
        "warning_checks": $WARNING_CHECKS,
        "failed_checks": $FAILED_CHECKS,
        "success_rate": $(( (TOTAL_CHECKS - FAILED_CHECKS) * 100 / TOTAL_CHECKS ))
    },
    "detailed_results": $results_json
}
EOF
    
    log_info "Health check report generated: $report_file"
    
    if [[ "$JSON_OUTPUT" == "true" ]]; then
        cat "$report_file"
    fi
}

# Display summary
display_summary() {
    if [[ "$JSON_OUTPUT" == "true" ]]; then
        return 0
    fi
    
    echo ""
    echo "============================================"
    echo "          HEALTH CHECK SUMMARY"
    echo "============================================"
    echo "Overall Health: $OVERALL_HEALTH"
    echo "Total Checks: $TOTAL_CHECKS"
    echo "Passed: $((TOTAL_CHECKS - FAILED_CHECKS - WARNING_CHECKS))"
    echo "Warnings: $WARNING_CHECKS"
    echo "Failed: $FAILED_CHECKS"
    echo "Success Rate: $(( (TOTAL_CHECKS - FAILED_CHECKS) * 100 / TOTAL_CHECKS ))%"
    echo "============================================"
    
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        echo -e "${RED}⚠️  Health check FAILED${NC}"
        echo "Please review the failed checks and take corrective action."
    elif [[ $WARNING_CHECKS -gt 0 ]]; then
        echo -e "${YELLOW}⚠️  Health check completed with WARNINGS${NC}"
        echo "Please review the warnings for potential issues."
    else
        echo -e "${GREEN}✅ All health checks PASSED${NC}"
        echo "System is operating normally."
    fi
    echo ""
}

# Main execution function
main() {
    setup_logging
    
    # Parse arguments
    parse_arguments "$@"
    
    log_info "Starting health check process"
    log_info "Model Type: $MODEL_TYPE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Timeout: ${TIMEOUT}s"
    
    # Determine which checks to run
    local checks_to_run
    if [[ -n "${SPECIFIC_CHECKS:-}" ]]; then
        IFS=',' read -ra checks_to_run <<< "$SPECIFIC_CHECKS"
    else
        checks_to_run=("aws" "sagemaker" "s3" "model" "inference" "performance" "monitoring" "security")
    fi
    
    # Run health checks
    for check in "${checks_to_run[@]}"; do
        case "$check" in
            "aws")
                check_aws_connectivity
                ;;
            "sagemaker")
                check_sagemaker_health
                ;;
            "s3")
                check_s3_model_artifacts
                ;;
            "model")
                check_s3_model_artifacts  # Alias for s3
                ;;
            "inference")
                check_model_inference
                ;;
            "performance")
                check_performance_benchmarks
                ;;
            "monitoring")
                check_monitoring_systems
                ;;
            "security")
                check_security_configurations
                ;;
            *)
                log_warn "Unknown health check: $check"
                ;;
        esac
    done
    
    # Generate report and display summary
    generate_health_report
    display_summary
    
    log_info "Health check completed - Session ID: $TIMESTAMP"
    
    # Exit with appropriate code
    if [[ "$OVERALL_HEALTH" == "unhealthy" ]]; then
        exit 1
    else
        exit 0
    fi
}

# Execute main function with all arguments
main "$@"
