#!/bin/bash

# BatteryMind Model Deployment Script
# Deploys models to various environments (local, AWS, edge)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_DIR="$PROJECT_ROOT/config"
MODELS_DIR="$PROJECT_ROOT/model-artifacts/trained_models"

# Default values
ENVIRONMENT="local"
MODEL_TYPE=""
MODEL_VERSION="latest"
DEPLOYMENT_TARGET="docker"
AWS_REGION="us-west-2"
ECR_REGISTRY=""
EDGE_DEVICE=""
DRY_RUN=false
FORCE_REBUILD=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
BatteryMind Model Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE              Model type (transformer|federated|rl|ensemble)
    -e, --environment ENV        Deployment environment (local|aws|edge)
    -v, --version VERSION        Model version (default: latest)
    -d, --target TARGET          Deployment target (docker|k8s|sagemaker|edge)
    -r, --region REGION          AWS region (default: us-west-2)
    --registry REGISTRY          ECR registry URL
    --edge-device DEVICE         Edge device identifier
    --dry-run                    Show what would be deployed without executing
    --force-rebuild              Force rebuild of containers
    -h, --help                   Show this help message

EXAMPLES:
    # Deploy transformer model locally
    $0 -t transformer -e local

    # Deploy to AWS SageMaker
    $0 -t ensemble -e aws -d sagemaker

    # Deploy to edge device
    $0 -t rl -e edge --edge-device battery-node-001

    # Dry run deployment
    $0 -t federated -e aws --dry-run
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                MODEL_TYPE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                MODEL_VERSION="$2"
                shift 2
                ;;
            -d|--target)
                DEPLOYMENT_TARGET="$2"
                shift 2
                ;;
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            --registry)
                ECR_REGISTRY="$2"
                shift 2
                ;;
            --edge-device)
                EDGE_DEVICE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate arguments
validate_args() {
    if [[ -z "$MODEL_TYPE" ]]; then
        log_error "Model type is required. Use -t or --type"
        exit 1
    fi

    case "$MODEL_TYPE" in
        transformer|federated|rl|ensemble)
            ;;
        *)
            log_error "Invalid model type: $MODEL_TYPE"
            log_error "Valid types: transformer, federated, rl, ensemble"
            exit 1
            ;;
    esac

    case "$ENVIRONMENT" in
        local|aws|edge)
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: local, aws, edge"
            exit 1
            ;;
    esac

    if [[ "$ENVIRONMENT" == "edge" && -z "$EDGE_DEVICE" ]]; then
        log_error "Edge device identifier is required for edge deployment"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi

    # Check AWS CLI for AWS deployments
    if [[ "$ENVIRONMENT" == "aws" ]] && ! command -v aws &> /dev/null; then
        log_error "AWS CLI is required for AWS deployments"
        exit 1
    fi

    # Check kubectl for Kubernetes deployments
    if [[ "$DEPLOYMENT_TARGET" == "k8s" ]] && ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required for Kubernetes deployments"
        exit 1
    fi

    # Check model files exist
    MODEL_DIR="$MODELS_DIR/${MODEL_TYPE}_${MODEL_VERSION}"
    if [[ ! -d "$MODEL_DIR" ]]; then
        log_error "Model directory not found: $MODEL_DIR"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Build Docker image
build_docker_image() {
    local dockerfile="$PROJECT_ROOT/deployment/containers/Dockerfile.$MODEL_TYPE"
    local image_name="batterymind-$MODEL_TYPE:$MODEL_VERSION"
    
    log_info "Building Docker image: $image_name"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build image: $image_name"
        return 0
    fi
    
    # Check if image exists and force rebuild is not set
    if [[ "$FORCE_REBUILD" == "false" ]] && docker image inspect "$image_name" &> /dev/null; then
        log_info "Image $image_name already exists. Use --force-rebuild to rebuild."
        return 0
    fi
    
    # Build image
    cd "$PROJECT_ROOT"
    docker build -f "$dockerfile" -t "$image_name" .
    
    log_success "Docker image built successfully: $image_name"
}

# Deploy to local environment
deploy_local() {
    local image_name="batterymind-$MODEL_TYPE:$MODEL_VERSION"
    local container_name="batterymind-$MODEL_TYPE-$MODEL_VERSION"
    
    log_info "Deploying to local environment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy container: $container_name"
        return 0
    fi
    
    # Stop existing container if running
    if docker ps -q -f name="$container_name" | grep -q .; then
        log_info "Stopping existing container: $container_name"
        docker stop "$container_name"
        docker rm "$container_name"
    fi
    
    # Run container
    docker run -d \
        --name "$container_name" \
        -p 8080:8080 \
        -v "$MODELS_DIR:/app/models:ro" \
        -v "$CONFIG_DIR:/app/config:ro" \
        -e MODEL_TYPE="$MODEL_TYPE" \
        -e MODEL_VERSION="$MODEL_VERSION" \
        "$image_name"
    
    # Wait for container to be ready
    log_info "Waiting for container to be ready..."
    for i in {1..30}; do
        if docker exec "$container_name" curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Container is ready and healthy"
            break
        fi
        sleep 2
    done
    
    log_success "Local deployment completed"
    log_info "Container: $container_name"
    log_info "URL: http://localhost:8080"
}

# Deploy to AWS
deploy_aws() {
    log_info "Deploying to AWS environment..."
    
    case "$DEPLOYMENT_TARGET" in
        docker)
            deploy_aws_ecs
            ;;
        k8s)
            deploy_aws_eks
            ;;
        sagemaker)
            deploy_aws_sagemaker
            ;;
        *)
            log_error "Unsupported AWS deployment target: $DEPLOYMENT_TARGET"
            exit 1
            ;;
    esac
}

# Deploy to AWS ECS
deploy_aws_ecs() {
    local image_name="batterymind-$MODEL_TYPE:$MODEL_VERSION"
    local ecr_image="$ECR_REGISTRY/$image_name"
    
    log_info "Deploying to AWS ECS..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy to ECS with image: $ecr_image"
        return 0
    fi
    
    # Push image to ECR
    push_to_ecr "$image_name" "$ecr_image"
    
    # Update ECS service
    local task_definition="batterymind-$MODEL_TYPE"
    local service_name="batterymind-$MODEL_TYPE-service"
    local cluster_name="batterymind-cluster"
    
    # Create or update task definition
    aws ecs register-task-definition \
        --family "$task_definition" \
        --requires-compatibilities FARGATE \
        --network-mode awsvpc \
        --cpu 512 \
        --memory 1024 \
        --execution-role-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/ecsTaskExecutionRole" \
        --container-definitions "[
            {
                \"name\": \"$MODEL_TYPE\",
                \"image\": \"$ecr_image\",
                \"portMappings\": [
                    {
                        \"containerPort\": 8080,
                        \"protocol\": \"tcp\"
                    }
                ],
                \"essential\": true,
                \"logConfiguration\": {
                    \"logDriver\": \"awslogs\",
                    \"options\": {
                        \"awslogs-group\": \"/ecs/batterymind\",
                        \"awslogs-region\": \"$AWS_REGION\",
                        \"awslogs-stream-prefix\": \"ecs\"
                    }
                }
            }
        ]" \
        --region "$AWS_REGION"
    
    # Update service
    aws ecs update-service \
        --cluster "$cluster_name" \
        --service "$service_name" \
        --task-definition "$task_definition" \
        --region "$AWS_REGION"
    
    log_success "AWS ECS deployment completed"
}

# Deploy to AWS SageMaker
deploy_aws_sagemaker() {
    log_info "Deploying to AWS SageMaker..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy to SageMaker endpoint"
        return 0
    fi
    
    # Use SageMaker deployment module
    python -m deployment.aws_sagemaker.model_deployment \
        --model-type "$MODEL_TYPE" \
        --model-version "$MODEL_VERSION" \
        --region "$AWS_REGION"
    
    log_success "AWS SageMaker deployment completed"
}

# Deploy to edge device
deploy_edge() {
    log_info "Deploying to edge device: $EDGE_DEVICE"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy to edge device: $EDGE_DEVICE"
        return 0
    fi
    
    # Build optimized image for edge
    local edge_dockerfile="$PROJECT_ROOT/deployment/containers/Dockerfile.edge"
    local edge_image="batterymind-$MODEL_TYPE-edge:$MODEL_VERSION"
    
    # Create edge-specific Dockerfile if it doesn't exist
    if [[ ! -f "$edge_dockerfile" ]]; then
        create_edge_dockerfile
    fi
    
    # Build edge image
    cd "$PROJECT_ROOT"
    docker build -f "$edge_dockerfile" -t "$edge_image" .
    
    # Save image as tar file
    local image_tar="/tmp/batterymind-$MODEL_TYPE-edge-$MODEL_VERSION.tar"
    docker save "$edge_image" -o "$image_tar"
    
    # Transfer to edge device (assuming SSH access)
    scp "$image_tar" "root@$EDGE_DEVICE:/tmp/"
    
    # Load and run on edge device
    ssh "root@$EDGE_DEVICE" << EOF
        docker load -i /tmp/$(basename "$image_tar")
        docker stop batterymind-$MODEL_TYPE || true
        docker rm batterymind-$MODEL_TYPE || true
        docker run -d \\
            --name batterymind-$MODEL_TYPE \\
            --restart unless-stopped \\
            -p 8080:8080 \\
            $edge_image
EOF
    
    # Cleanup
    rm "$image_tar"
    
    log_success "Edge deployment completed"
    log_info "Edge device: $EDGE_DEVICE"
    log_info "SSH into device and check: docker ps"
}

# Push image to ECR
push_to_ecr() {
    local local_image="$1"
    local ecr_image="$2"
    
    log_info "Pushing image to ECR: $ecr_image"
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_REGISTRY"
    
    # Tag and push
    docker tag "$local_image" "$ecr_image"
    docker push "$ecr_image"
    
    log_success "Image pushed to ECR"
}

# Create edge-specific Dockerfile
create_edge_dockerfile() {
    local edge_dockerfile="$PROJECT_ROOT/deployment/containers/Dockerfile.edge"
    
    cat > "$edge_dockerfile" << 'EOF'
# BatteryMind Edge Deployment
FROM python:3.9-alpine

# Install minimal dependencies
RUN apk add --no-cache gcc musl-dev

WORKDIR /app

# Copy only necessary files
COPY requirements-edge.txt .
RUN pip install --no-cache-dir -r requirements-edge.txt

COPY inference/ ./inference/
COPY deployment/edge_deployment/ ./deployment/edge_deployment/

# Create minimal user
RUN adduser -D batterymind
USER batterymind

EXPOSE 8080

CMD ["python", "-m", "deployment.edge_deployment.edge_runtime"]
EOF
    
    log_info "Created edge-specific Dockerfile"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add cleanup logic here
}

# Main execution
main() {
    trap cleanup EXIT
    
    log_info "Starting BatteryMind model deployment..."
    
    parse_args "$@"
    validate_args
    check_prerequisites
    
    # Build Docker image
    build_docker_image
    
    # Deploy based on environment
    case "$ENVIRONMENT" in
        local)
            deploy_local
            ;;
        aws)
            deploy_aws
            ;;
        edge)
            deploy_edge
            ;;
    esac
    
    log_success "Deployment completed successfully!"
    
    # Show deployment information
    cat << EOF

Deployment Summary:
==================
Model Type:     $MODEL_TYPE
Version:        $MODEL_VERSION
Environment:    $ENVIRONMENT
Target:         $DEPLOYMENT_TARGET
$([ "$ENVIRONMENT" == "edge" ] && echo "Edge Device:    $EDGE_DEVICE")

Next Steps:
-----------
1. Verify deployment health
2. Run integration tests
3. Monitor performance metrics
4. Set up alerting

EOF
}

# Run main function with all arguments
main "$@"
