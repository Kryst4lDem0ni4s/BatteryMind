#!/bin/bash

# BatteryMind Model Rollback Script
# Safe rollback capabilities for model deployments
#
# Author: BatteryMind Development Team
# Version: 1.0.0
# Dependencies: AWS CLI, Docker

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_FILE="/var/log/batterymind/model_rollback.log"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# AWS Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_MODEL_BUCKET="${S3_MODEL_BUCKET:-batterymind-models}"
SAGEMAKER_ENDPOINT_PREFIX="${SAGEMAKER_ENDPOINT_PREFIX:-batterymind}"

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

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Setup logging
setup_logging() {
    local log_dir
    log_dir="$(dirname "$LOG_FILE")"
    
    if [[ ! -d "$log_dir" ]]; then
        sudo mkdir -p "$log_dir" || {
            LOG_FILE="./model_rollback_${TIMESTAMP}.log"
            log_warn "Cannot create system log directory, using local log file: $LOG_FILE"
        }
    fi
    
    log_info "Model rollback started - Session ID: $TIMESTAMP"
}

# Display usage information
usage() {
    cat << EOF
BatteryMind Model Rollback Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE           Model type (transformer, federated, rl, ensemble)
    -e, --env ENVIRONMENT     Target environment (dev, staging, prod)
    -b, --backup-dir DIR      Backup directory path
    -v, --target-version VER  Target version to rollback to
    -l, --list-backups        List available backups
    -f, --force              Skip confirmation prompts
    -d, --dry-run            Show what would be done without executing
    -h, --help               Show this help message

EXAMPLES:
    $0 -t transformer -e prod --backup-dir /var/backups/batterymind/20240115_143022
    $0 --type rl --env staging --target-version 1.5.0
    $0 -l -t transformer -e prod

EOF
}

# Parse command line arguments
parse_arguments() {
    PARSED_ARGS=$(getopt -o t:e:b:v:lfdh --long type:,env:,backup-dir:,target-version:,list-backups,force,dry-run,help -n "$0" -- "$@")
    
    if [[ $? -ne 0 ]]; then
        error_exit "Invalid arguments provided"
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
            -b|--backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -v|--target-version)
                TARGET_VERSION="$2"
                shift 2
                ;;
            -l|--list-backups)
                LIST_BACKUPS=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
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
                error_exit "Unknown option: $1"
                ;;
        esac
    done
    
    # Set defaults
    FORCE="${FORCE:-false}"
    DRY_RUN="${DRY_RUN:-false}"
    LIST_BACKUPS="${LIST_BACKUPS:-false}"
}

# List available backups
list_available_backups() {
    log_info "Listing available backups..."
    
    local backup_base="/var/backups/batterymind"
    local s3_backup_prefix="s3://$S3_MODEL_BUCKET/backups/"
    
    echo -e "\n${BLUE}Local Backups:${NC}"
    if [[ -d "$backup_base" ]]; then
        find "$backup_base" -name "backup_metadata.json" | while read -r metadata_file; do
            local backup_dir
            backup_dir="$(dirname "$metadata_file")"
            local backup_name
            backup_name="$(basename "$backup_dir")"
            
            if [[ -f "$metadata_file" ]]; then
                local model_type env timestamp
                model_type=$(jq -r '.model_type // "unknown"' "$metadata_file")
                env=$(jq -r '.environment // "unknown"' "$metadata_file")
                timestamp=$(jq -r '.timestamp // "unknown"' "$metadata_file")
                
                if [[ -z "${MODEL_TYPE:-}" || "$model_type" == "$MODEL_TYPE" ]]; then
                    if [[ -z "${ENVIRONMENT:-}" || "$env" == "$ENVIRONMENT" ]]; then
                        echo "  $backup_name - $model_type ($env) - $timestamp"
                    fi
                fi
            fi
        done
    else
        echo "  No local backups found"
    fi
    
    echo -e "\n${BLUE}S3 Model Versions:${NC}"
    if [[ -n "${MODEL_TYPE:-}" ]]; then
        aws s3 ls "s3://$S3_MODEL_BUCKET/models/${MODEL_TYPE}/" --recursive | grep "version.txt" | while read -r line; do
            local version_path
            version_path=$(echo "$line" | awk '{print $4}')
            local version_dir
            version_dir=$(dirname "$version_path" | sed "s/models\/${MODEL_TYPE}\///")
            
            if [[ "$version_dir" != "current" ]]; then
                local version_content
                version_content=$(aws s3 cp "s3://$S3_MODEL_BUCKET/$version_path" - 2>/dev/null || echo "unknown")
                echo "  Version: $version_dir ($version_content)"
            fi
        done
    else
        echo "  Specify model type to list S3 versions"
    fi
    
    echo ""
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check required parameters
    if [[ -z "${MODEL_TYPE:-}" ]]; then
        error_exit "Model type is required. Use -t/--type to specify."
    fi
    
    if [[ -z "${ENVIRONMENT:-}" ]]; then
        error_exit "Environment is required. Use -e/--env to specify."
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error_exit "AWS credentials not configured properly"
    fi
    
    # Validate model type
    local valid_types=("transformer" "federated" "rl" "ensemble")
    if [[ ! " ${valid_types[*]} " =~ " ${MODEL_TYPE} " ]]; then
        error_exit "Invalid model type: $MODEL_TYPE. Valid types: ${valid_types[*]}"
    fi
    
    # Validate environment
    local valid_envs=("dev" "staging" "prod")
    if [[ ! " ${valid_envs[*]} " =~ " ${ENVIRONMENT} " ]]; then
        error_exit "Invalid environment: $ENVIRONMENT. Valid environments: ${valid_envs[*]}"
    fi
    
    log_info "Prerequisites validation completed"
}

# Find backup directory
find_backup_directory() {
    if [[ -n "${BACKUP_DIR:-}" ]]; then
        if [[ ! -d "$BACKUP_DIR" ]]; then
            error_exit "Specified backup directory does not exist: $BACKUP_DIR"
        fi
        return 0
    fi
    
    # Find most recent backup
    local backup_base="/var/backups/batterymind"
    local latest_backup
    
    if [[ -d "$backup_base" ]]; then
        latest_backup=$(find "$backup_base" -name "backup_metadata.json" -exec grep -l "\"model_type\": \"$MODEL_TYPE\"" {} \; -exec grep -l "\"environment\": \"$ENVIRONMENT\"" {} \; | head -1)
        
        if [[ -n "$latest_backup" ]]; then
            BACKUP_DIR="$(dirname "$latest_backup")"
            log_info "Using latest backup: $BACKUP_DIR"
        else
            error_exit "No suitable backup found for $MODEL_TYPE in $ENVIRONMENT environment"
        fi
    else
        error_exit "No backup directory found: $backup_base"
    fi
}

# Validate backup
validate_backup() {
    log_info "Validating backup: $BACKUP_DIR"
    
    local metadata_file="$BACKUP_DIR/backup_metadata.json"
    if [[ ! -f "$metadata_file" ]]; then
        error_exit "Backup metadata not found: $metadata_file"
    fi
    
    # Validate backup metadata
    local backup_model_type backup_env
    backup_model_type=$(jq -r '.model_type // "unknown"' "$metadata_file")
    backup_env=$(jq -r '.environment // "unknown"' "$metadata_file")
    
    if [[ "$backup_model_type" != "$MODEL_TYPE" ]]; then
        error_exit "Backup model type mismatch. Expected: $MODEL_TYPE, Found: $backup_model_type"
    fi
    
    if [[ "$backup_env" != "$ENVIRONMENT" ]]; then
        error_exit "Backup environment mismatch. Expected: $ENVIRONMENT, Found: $backup_env"
    fi
    
    # Check S3 backup
    local s3_backup_path
    s3_backup_path=$(jq -r '.s3_backup_path // ""' "$metadata_file")
    if [[ -n "$s3_backup_path" ]]; then
        if ! aws s3 ls "$s3_backup_path" &> /dev/null; then
            log_warn "S3 backup path not accessible: $s3_backup_path"
        else
            log_info "S3 backup validated: $s3_backup_path"
        fi
    fi
    
    log_info "Backup validation completed successfully"
}

# Restore SageMaker endpoint
restore_sagemaker_endpoint() {
    log_info "Restoring SageMaker endpoint configuration..."
    
    local endpoint_config_file="$BACKUP_DIR/endpoint_full_config.json"
    local endpoint_name="${SAGEMAKER_ENDPOINT_PREFIX}-${MODEL_TYPE}-${ENVIRONMENT}"
    
    if [[ ! -f "$endpoint_config_file" ]]; then
        log_warn "Endpoint configuration backup not found, skipping SageMaker restore"
        return 0
    fi
    
    # Extract configuration details
    local config_name model_name
    config_name=$(jq -r '.EndpointConfigName' "$endpoint_config_file")
    model_name=$(jq -r '.ProductionVariants[0].ModelName' "$endpoint_config_file")
    
    # Create rollback model and configuration names
    local rollback_model_name="${endpoint_name}-rollback-${TIMESTAMP}"
    local rollback_config_name="${endpoint_name}-rollback-config-${TIMESTAMP}"
    
    # Restore model artifacts from S3 backup
    local s3_backup_path
    s3_backup_path=$(jq -r '.s3_backup_path // ""' "$BACKUP_DIR/backup_metadata.json")
    
    if [[ -n "$s3_backup_path" ]]; then
        log_info "Restoring model artifacts from S3 backup..."
        
        # Determine Docker image based on model type
        local docker_image
        case "$MODEL_TYPE" in
            "transformer") docker_image="batterymind/transformer:latest" ;;
            "federated") docker_image="batterymind/federated:latest" ;;
            "rl") docker_image="batterymind/rl:latest" ;;
            "ensemble") docker_image="batterymind/ensemble:latest" ;;
        esac
        
        # Create rollback model
        aws sagemaker create-model \
            --model-name "$rollback_model_name" \
            --primary-container Image="$docker_image",ModelDataUrl="$s3_backup_path" \
            --execution-role-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/SageMakerExecutionRole" \
            --region "$AWS_REGION" || error_exit "Failed to create rollback model"
        
        # Create rollback endpoint configuration
        aws sagemaker create-endpoint-config \
            --endpoint-config-name "$rollback_config_name" \
            --production-variants VariantName=primary,ModelName="$rollback_model_name",InitialInstanceCount=1,InstanceType=ml.m5.large,InitialVariantWeight=1 \
            --region "$AWS_REGION" || error_exit "Failed to create rollback endpoint configuration"
        
        # Update endpoint to rollback configuration
        aws sagemaker update-endpoint \
            --endpoint-name "$endpoint_name" \
            --endpoint-config-name "$rollback_config_name" \
            --region "$AWS_REGION" || error_exit "Failed to update endpoint to rollback configuration"
        
        log_info "SageMaker endpoint rollback initiated"
    fi
}

# Restore S3 model artifacts
restore_s3_artifacts() {
    log_info "Restoring S3 model artifacts..."
    
    local s3_backup_path
    s3_backup_path=$(jq -r '.s3_backup_path // ""' "$BACKUP_DIR/backup_metadata.json")
    
    if [[ -z "$s3_backup_path" ]]; then
        log_warn "S3 backup path not found in metadata, skipping S3 restore"
        return 0
    fi
    
    local production_path="s3://$S3_MODEL_BUCKET/models/${MODEL_TYPE}/current/"
    
    # Create backup of current state before restore
    local pre_rollback_backup="s3://$S3_MODEL_BUCKET/pre-rollback-backup/${MODEL_TYPE}/${TIMESTAMP}/"
    aws s3 sync "$production_path" "$pre_rollback_backup" --quiet
    
    # Restore from backup
    aws s3 sync "$s3_backup_path" "$production_path" --delete || error_exit "Failed to restore S3 artifacts"
    
    log_info "S3 model artifacts restored successfully"
    log_info "Pre-rollback backup created at: $pre_rollback_backup"
}

# Wait for rollback completion
wait_for_rollback_completion() {
    log_info "Waiting for rollback to complete..."
    
    local endpoint_name="${SAGEMAKER_ENDPOINT_PREFIX}-${MODEL_TYPE}-${ENVIRONMENT}"
    local timeout=600  # 10 minutes
    local elapsed=0
    local status
    
    while [[ $elapsed -lt $timeout ]]; do
        status=$(aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" --region "$AWS_REGION" --query 'EndpointStatus' --output text 2>/dev/null || echo "NotFound")
        
        case "$status" in
            "InService")
                log_info "Rollback completed successfully"
                return 0
                ;;
            "Updating")
                log_info "Rollback in progress... (${elapsed}s elapsed)"
                sleep 30
                elapsed=$((elapsed + 30))
                ;;
            "Failed"|"OutOfService")
                error_exit "Rollback failed with status: $status"
                ;;
            "NotFound")
                log_warn "Endpoint not found, rollback may not be applicable"
                return 0
                ;;
            *)
                log_warn "Unknown rollback status: $status"
                sleep 30
                elapsed=$((elapsed + 30))
                ;;
        esac
    done
    
    error_exit "Rollback timeout after ${timeout} seconds"
}

# Run post-rollback health checks
run_post_rollback_checks() {
    log_info "Running post-rollback health checks..."
    
    # Run health check script
    if ! "$SCRIPT_DIR/health_check.sh" --model-type "$MODEL_TYPE" --environment "$ENVIRONMENT" --timeout 120; then
        error_exit "Post-rollback health checks failed"
    fi
    
    log_info "Post-rollback health checks passed"
}

# Generate rollback report
generate_rollback_report() {
    log_info "Generating rollback report..."
    
    local report_file="/var/log/batterymind/rollback_report_${TIMESTAMP}.json"
    local backup_metadata
    backup_metadata=$(cat "$BACKUP_DIR/backup_metadata.json")
    
    cat > "$report_file" << EOF
{
    "rollback_session_id": "$TIMESTAMP",
    "model_type": "$MODEL_TYPE",
    "environment": "$ENVIRONMENT",
    "rollback_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "backup_used": "$BACKUP_DIR",
    "backup_metadata": $backup_metadata,
    "rollback_successful": true,
    "performed_by": "$(whoami)",
    "actions_performed": [
        "SageMaker endpoint rollback",
        "S3 artifacts restoration",
        "Health checks validation"
    ]
}
EOF
    
    log_info "Rollback report generated: $report_file"
}

# Send notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # SNS notification (if configured)
    if [[ -n "${SNS_TOPIC_ARN:-}" ]]; then
        aws sns publish \
            --topic-arn "$SNS_TOPIC_ARN" \
            --message "BatteryMind Model Rollback - $status: $message" \
            --subject "Model Rollback Notification" \
            --region "$AWS_REGION" 2>/dev/null || log_warn "Failed to send SNS notification"
    fi
}

# Main execution function
main() {
    setup_logging
    
    # Parse arguments
    parse_arguments "$@"
    
    # Handle list backups request
    if [[ "$LIST_BACKUPS" == "true" ]]; then
        list_available_backups
        exit 0
    fi
    
    log_info "Starting model rollback process"
    log_info "Model Type: $MODEL_TYPE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Backup Directory: ${BACKUP_DIR:-auto-detect}"
    log_info "Target Version: ${TARGET_VERSION:-from backup}"
    log_info "Dry Run: $DRY_RUN"
    
    # Validate prerequisites
    validate_prerequisites
    
    # Find backup directory if not specified
    find_backup_directory
    
    # Validate backup
    validate_backup
    
    # Show confirmation prompt unless forced
    if [[ "$FORCE" != "true" && "$DRY_RUN" != "true" ]]; then
        echo -e "${YELLOW}Warning:${NC} You are about to rollback $MODEL_TYPE model in $ENVIRONMENT environment."
        echo "This will restore the system to a previous state and may result in data loss."
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Rollback cancelled by user"
            exit 0
        fi
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run completed successfully - no actual rollback performed"
        log_info "Would restore from backup: $BACKUP_DIR"
        exit 0
    fi
    
    # Perform rollback
    restore_s3_artifacts
    restore_sagemaker_endpoint
    wait_for_rollback_completion
    run_post_rollback_checks
    generate_rollback_report
    
    # Send success notification
    send_notification "SUCCESS" "Model $MODEL_TYPE successfully rolled back in $ENVIRONMENT environment"
    
    log_info "Model rollback completed successfully"
    log_info "Rollback session ID: $TIMESTAMP"
}

# Execute main function with all arguments
main "$@"
