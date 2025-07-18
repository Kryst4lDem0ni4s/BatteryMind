#!/bin/bash

# BatteryMind Model Update Script
# Automated model deployment with validation and rollback capabilities
#
# Author: BatteryMind Development Team
# Version: 1.0.0
# Dependencies: AWS CLI, Docker, kubectl (optional)

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_FILE="/var/log/batterymind/model_update.log"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_MODEL_TYPE="transformer"
DEFAULT_ENVIRONMENT="staging"
DEFAULT_VALIDATION_TIMEOUT=300
DEFAULT_ROLLBACK_ON_FAILURE=true
DEFAULT_BACKUP_RETENTION_DAYS=30

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

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
    fi
}

# Error handling
error_exit() {
    log_error "$1"
    if [[ "${ROLLBACK_ON_FAILURE:-$DEFAULT_ROLLBACK_ON_FAILURE}" == "true" ]]; then
        log_info "Initiating automatic rollback..."
        rollback_deployment
    fi
    exit 1
}

# Setup logging directory
setup_logging() {
    local log_dir
    log_dir="$(dirname "$LOG_FILE")"
    
    if [[ ! -d "$log_dir" ]]; then
        sudo mkdir -p "$log_dir" || {
            LOG_FILE="./model_update_${TIMESTAMP}.log"
            log_warn "Cannot create system log directory, using local log file: $LOG_FILE"
        }
    fi
    
    log_info "Model update started - Session ID: $TIMESTAMP"
}

# Display usage information
usage() {
    cat << EOF
BatteryMind Model Update Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE           Model type (transformer, federated, rl, ensemble)
    -v, --version VERSION     Model version to deploy
    -e, --env ENVIRONMENT     Target environment (dev, staging, prod)
    -s, --source SOURCE       Source path for model artifacts
    -b, --bucket BUCKET       S3 bucket for model storage
    -r, --region REGION       AWS region
    -n, --no-rollback         Disable automatic rollback on failure
    -d, --dry-run            Perform validation without actual deployment
    -f, --force              Skip validation prompts
    -h, --help               Show this help message

EXAMPLES:
    $0 -t transformer -v 1.2.0 -e staging
    $0 --type rl --version 2.0.0 --env prod --source ./models/rl_v2
    $0 -t ensemble -v 1.5.0 -e prod --dry-run

ENVIRONMENT VARIABLES:
    AWS_REGION               AWS region (default: us-east-1)
    S3_MODEL_BUCKET          S3 bucket for models (default: batterymind-models)
    SAGEMAKER_ENDPOINT_PREFIX Endpoint prefix (default: batterymind)
    DEBUG                    Enable debug logging (default: false)
    ROLLBACK_ON_FAILURE      Auto-rollback on failure (default: true)

EOF
}

# Parse command line arguments
parse_arguments() {
    PARSED_ARGS=$(getopt -o t:v:e:s:b:r:ndfh --long type:,version:,env:,source:,bucket:,region:,no-rollback,dry-run,force,help -n "$0" -- "$@")
    
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
            -v|--version)
                MODEL_VERSION="$2"
                shift 2
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--source)
                SOURCE_PATH="$2"
                shift 2
                ;;
            -b|--bucket)
                S3_MODEL_BUCKET="$2"
                shift 2
                ;;
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -n|--no-rollback)
                ROLLBACK_ON_FAILURE=false
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
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
    MODEL_TYPE="${MODEL_TYPE:-$DEFAULT_MODEL_TYPE}"
    ENVIRONMENT="${ENVIRONMENT:-$DEFAULT_ENVIRONMENT}"
    ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-$DEFAULT_ROLLBACK_ON_FAILURE}"
    DRY_RUN="${DRY_RUN:-false}"
    FORCE="${FORCE:-false}"
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check required tools
    local required_tools=("aws" "docker" "python3" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "Required tool '$tool' is not installed"
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error_exit "AWS credentials not configured properly"
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running"
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
    
    log_info "Prerequisites validation completed successfully"
}

# Backup current deployment
backup_current_deployment() {
    log_info "Creating backup of current deployment..."
    
    local backup_dir="/var/backups/batterymind/${TIMESTAMP}"
    local endpoint_name="${SAGEMAKER_ENDPOINT_PREFIX}-${MODEL_TYPE}-${ENVIRONMENT}"
    
    # Create backup directory
    sudo mkdir -p "$backup_dir" || error_exit "Failed to create backup directory"
    
    # Backup SageMaker endpoint configuration
    if aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" &> /dev/null; then
        aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" > "$backup_dir/endpoint_config.json"
        aws sagemaker describe-endpoint-config --endpoint-config-name "${endpoint_name}-config" > "$backup_dir/endpoint_full_config.json"
        log_info "SageMaker endpoint configuration backed up"
    fi
    
    # Backup current model artifacts from S3
    local s3_backup_path="s3://$S3_MODEL_BUCKET/backups/${MODEL_TYPE}/${TIMESTAMP}/"
    aws s3 sync "s3://$S3_MODEL_BUCKET/models/${MODEL_TYPE}/current/" "$s3_backup_path" --quiet
    
    # Save backup metadata
    cat > "$backup_dir/backup_metadata.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "model_type": "$MODEL_TYPE",
    "environment": "$ENVIRONMENT",
    "s3_backup_path": "$s3_backup_path",
    "endpoint_name": "$endpoint_name",
    "created_by": "$(whoami)",
    "source_version": "$(get_current_model_version)"
}
EOF
    
    log_info "Backup completed: $backup_dir"
    echo "$backup_dir" > "/tmp/batterymind_last_backup_${MODEL_TYPE}_${ENVIRONMENT}"
}

# Get current model version
get_current_model_version() {
    local version_file="s3://$S3_MODEL_BUCKET/models/${MODEL_TYPE}/current/version.txt"
    aws s3 cp "$version_file" - 2>/dev/null || echo "unknown"
}

# Validate model artifacts
validate_model_artifacts() {
    log_info "Validating model artifacts..."
    
    if [[ -z "${SOURCE_PATH:-}" ]]; then
        error_exit "Source path not specified. Use -s/--source to specify model artifacts location"
    fi
    
    if [[ ! -d "$SOURCE_PATH" ]]; then
        error_exit "Source path does not exist: $SOURCE_PATH"
    fi
    
    # Check required files based on model type
    local required_files
    case "$MODEL_TYPE" in
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
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$SOURCE_PATH/$file" ]]; then
            error_exit "Required file missing: $file in $SOURCE_PATH"
        fi
    done
    
    # Validate model integrity
    log_info "Running model integrity checks..."
    python3 "$SCRIPT_DIR/validate_model.py" --model-type "$MODEL_TYPE" --source "$SOURCE_PATH" || error_exit "Model validation failed"
    
    log_info "Model artifacts validation completed successfully"
}

# Upload model to S3
upload_model_to_s3() {
    log_info "Uploading model artifacts to S3..."
    
    local s3_path="s3://$S3_MODEL_BUCKET/models/${MODEL_TYPE}/${MODEL_VERSION}/"
    local staging_path="s3://$S3_MODEL_BUCKET/staging/${MODEL_TYPE}/${MODEL_VERSION}/"
    
    # Upload to staging first
    aws s3 sync "$SOURCE_PATH" "$staging_path" --delete || error_exit "Failed to upload to S3 staging"
    
    # Create version metadata
    cat > "/tmp/version_metadata_${TIMESTAMP}.json" << EOF
{
    "version": "$MODEL_VERSION",
    "model_type": "$MODEL_TYPE",
    "upload_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "source_path": "$SOURCE_PATH",
    "environment": "$ENVIRONMENT",
    "uploaded_by": "$(whoami)"
}
EOF
    
    aws s3 cp "/tmp/version_metadata_${TIMESTAMP}.json" "${staging_path}metadata.json"
    echo "$MODEL_VERSION" > "/tmp/version_${TIMESTAMP}.txt"
    aws s3 cp "/tmp/version_${TIMESTAMP}.txt" "${staging_path}version.txt"
    
    log_info "Model uploaded to S3 staging: $staging_path"
    
    # Clean up temporary files
    rm -f "/tmp/version_metadata_${TIMESTAMP}.json" "/tmp/version_${TIMESTAMP}.txt"
}

# Deploy to SageMaker
deploy_to_sagemaker() {
    log_info "Deploying model to SageMaker..."
    
    local endpoint_name="${SAGEMAKER_ENDPOINT_PREFIX}-${MODEL_TYPE}-${ENVIRONMENT}"
    local model_name="${endpoint_name}-model-${MODEL_VERSION}-${TIMESTAMP}"
    local config_name="${endpoint_name}-config-${MODEL_VERSION}-${TIMESTAMP}"
    local s3_model_path="s3://$S3_MODEL_BUCKET/staging/${MODEL_TYPE}/${MODEL_VERSION}/"
    
    # Create SageMaker model
    local docker_image
    case "$MODEL_TYPE" in
        "transformer")
            docker_image="batterymind/transformer:latest"
            ;;
        "federated")
            docker_image="batterymind/federated:latest"
            ;;
        "rl")
            docker_image="batterymind/rl:latest"
            ;;
        "ensemble")
            docker_image="batterymind/ensemble:latest"
            ;;
    esac
    
    aws sagemaker create-model \
        --model-name "$model_name" \
        --primary-container Image="$docker_image",ModelDataUrl="$s3_model_path" \
        --execution-role-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/SageMakerExecutionRole" \
        --region "$AWS_REGION" || error_exit "Failed to create SageMaker model"
    
    # Create endpoint configuration
    aws sagemaker create-endpoint-config \
        --endpoint-config-name "$config_name" \
        --production-variants VariantName=primary,ModelName="$model_name",InitialInstanceCount=1,InstanceType=ml.m5.large,InitialVariantWeight=1 \
        --region "$AWS_REGION" || error_exit "Failed to create endpoint configuration"
    
    # Check if endpoint exists
    if aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" --region "$AWS_REGION" &> /dev/null; then
        log_info "Updating existing endpoint: $endpoint_name"
        aws sagemaker update-endpoint \
            --endpoint-name "$endpoint_name" \
            --endpoint-config-name "$config_name" \
            --region "$AWS_REGION" || error_exit "Failed to update endpoint"
    else
        log_info "Creating new endpoint: $endpoint_name"
        aws sagemaker create-endpoint \
            --endpoint-name "$endpoint_name" \
            --endpoint-config-name "$config_name" \
            --region "$AWS_REGION" || error_exit "Failed to create endpoint"
    fi
    
    log_info "SageMaker deployment initiated"
}

# Wait for deployment completion
wait_for_deployment() {
    log_info "Waiting for deployment to complete..."
    
    local endpoint_name="${SAGEMAKER_ENDPOINT_PREFIX}-${MODEL_TYPE}-${ENVIRONMENT}"
    local timeout="${VALIDATION_TIMEOUT:-$DEFAULT_VALIDATION_TIMEOUT}"
    local elapsed=0
    local status
    
    while [[ $elapsed -lt $timeout ]]; do
        status=$(aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" --region "$AWS_REGION" --query 'EndpointStatus' --output text)
        
        case "$status" in
            "InService")
                log_info "Deployment completed successfully"
                return 0
                ;;
            "Creating"|"Updating")
                log_info "Deployment in progress... (${elapsed}s elapsed)"
                sleep 30
                elapsed=$((elapsed + 30))
                ;;
            "Failed"|"OutOfService")
                error_exit "Deployment failed with status: $status"
                ;;
            *)
                log_warn "Unknown deployment status: $status"
                sleep 30
                elapsed=$((elapsed + 30))
                ;;
        esac
    done
    
    error_exit "Deployment timeout after ${timeout} seconds"
}

# Run health checks
run_health_checks() {
    log_info "Running post-deployment health checks..."
    
    # Run comprehensive health check
    if ! "$SCRIPT_DIR/health_check.sh" --model-type "$MODEL_TYPE" --environment "$ENVIRONMENT" --timeout 120; then
        error_exit "Health checks failed"
    fi
    
    log_info "Health checks passed successfully"
}

# Promote staging to production
promote_to_production() {
    log_info "Promoting staging deployment to production..."
    
    local staging_path="s3://$S3_MODEL_BUCKET/staging/${MODEL_TYPE}/${MODEL_VERSION}/"
    local production_path="s3://$S3_MODEL_BUCKET/models/${MODEL_TYPE}/current/"
    local versioned_path="s3://$S3_MODEL_BUCKET/models/${MODEL_TYPE}/${MODEL_VERSION}/"
    
    # Copy from staging to versioned location
    aws s3 sync "$staging_path" "$versioned_path" --delete
    
    # Update current production pointer
    aws s3 sync "$staging_path" "$production_path" --delete
    
    log_info "Model promoted to production successfully"
}

# Rollback deployment
rollback_deployment() {
    log_warn "Initiating deployment rollback..."
    
    local backup_file="/tmp/batterymind_last_backup_${MODEL_TYPE}_${ENVIRONMENT}"
    if [[ ! -f "$backup_file" ]]; then
        log_error "No backup information found for rollback"
        return 1
    fi
    
    local backup_dir
    backup_dir=$(cat "$backup_file")
    
    if [[ ! -f "$backup_dir/backup_metadata.json" ]]; then
        log_error "Backup metadata not found: $backup_dir/backup_metadata.json"
        return 1
    fi
    
    # Execute rollback script
    "$SCRIPT_DIR/rollback_model.sh" --backup-dir "$backup_dir" --model-type "$MODEL_TYPE" --environment "$ENVIRONMENT"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    local backup_base="/var/backups/batterymind"
    local retention_days="${BACKUP_RETENTION_DAYS:-$DEFAULT_BACKUP_RETENTION_DAYS}"
    
    if [[ -d "$backup_base" ]]; then
        find "$backup_base" -type d -name "20*" -mtime "+$retention_days" -exec rm -rf {} + 2>/dev/null || true
    fi
    
    # Cleanup S3 backups
    local s3_backup_base="s3://$S3_MODEL_BUCKET/backups/${MODEL_TYPE}/"
    aws s3api list-objects-v2 --bucket "$S3_MODEL_BUCKET" --prefix "backups/${MODEL_TYPE}/" --query 'Contents[?LastModified<=`'"$(date -d "$retention_days days ago" -u +%Y-%m-%dT%H:%M:%SZ)"'`].Key' --output text | while read -r key; do
        if [[ -n "$key" ]]; then
            aws s3 rm "s3://$S3_MODEL_BUCKET/$key"
        fi
    done
    
    log_info "Backup cleanup completed"
}

# Send notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # SNS notification (if configured)
    if [[ -n "${SNS_TOPIC_ARN:-}" ]]; then
        aws sns publish \
            --topic-arn "$SNS_TOPIC_ARN" \
            --message "BatteryMind Model Update - $status: $message" \
            --subject "Model Update Notification" \
            --region "$AWS_REGION" 2>/dev/null || log_warn "Failed to send SNS notification"
    fi
    
    # Slack notification (if configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color
        case "$status" in
            "SUCCESS") color="good" ;;
            "FAILURE") color="danger" ;;
            *) color="warning" ;;
        esac
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"$color\",\"title\":\"BatteryMind Model Update\",\"text\":\"$message\",\"fields\":[{\"title\":\"Model Type\",\"value\":\"$MODEL_TYPE\",\"short\":true},{\"title\":\"Environment\",\"value\":\"$ENVIRONMENT\",\"short\":true},{\"title\":\"Version\",\"value\":\"${MODEL_VERSION:-unknown}\",\"short\":true}]}]}" \
            "$SLACK_WEBHOOK_URL" 2>/dev/null || log_warn "Failed to send Slack notification"
    fi
}

# Main execution function
main() {
    setup_logging
    
    # Parse arguments
    parse_arguments "$@"
    
    log_info "Starting model update process"
    log_info "Model Type: $MODEL_TYPE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: ${MODEL_VERSION:-not specified}"
    log_info "Source: ${SOURCE_PATH:-not specified}"
    log_info "Dry Run: $DRY_RUN"
    
    # Validate prerequisites
    validate_prerequisites
    
    # Show confirmation prompt unless forced
    if [[ "$FORCE" != "true" && "$DRY_RUN" != "true" ]]; then
        echo -e "${YELLOW}Warning:${NC} You are about to deploy $MODEL_TYPE model to $ENVIRONMENT environment."
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Validate model artifacts
    if [[ -n "${SOURCE_PATH:-}" ]]; then
        validate_model_artifacts
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run completed successfully - no actual deployment performed"
        exit 0
    fi
    
    # Backup current deployment
    backup_current_deployment
    
    # Upload model to S3
    if [[ -n "${SOURCE_PATH:-}" ]]; then
        upload_model_to_s3
    fi
    
    # Deploy to SageMaker
    deploy_to_sagemaker
    
    # Wait for deployment completion
    wait_for_deployment
    
    # Run health checks
    run_health_checks
    
    # Promote to production if staging deployment
    if [[ "$ENVIRONMENT" == "staging" ]]; then
        promote_to_production
    fi
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Send success notification
    send_notification "SUCCESS" "Model $MODEL_TYPE version ${MODEL_VERSION:-latest} successfully deployed to $ENVIRONMENT"
    
    log_info "Model update completed successfully"
    log_info "Deployment session ID: $TIMESTAMP"
}

# Execute main function with all arguments
main "$@"
