ARG RESOURCE_BUNDLE_VERSION=latest
# If Exported to Plainsight Prod from protege
# From us-west1-docker.pkg.dev/plainsightai-prod/oci/filter-huggingface-vision-model:${RESOURCE_BUNDLE_VERSION} as model
# If Exported to Planisight Dev from protege
# From us-central1-docker.pkg.dev/plainsightai-dev/oci/filter-huggingface-vision-model:${RESOURCE_BUNDLE_VERSION} as model
FROM us-west1-docker.pkg.dev/plainsightai-prod/oci/filter_base:python-3.11

# Copy both models and entrypoint script from the model image
# COPY --from=model /app/models /app/models
# COPY --from=model /app/entrypoint.sh /app/entrypoint.sh

# Transformers cache to be deprecated and HF_HOME to be used soon... 
# ENV TRANSFORMERS_CACHE=/app/models/hfcache 
# ENV HF_HOME=/app/models/hfcache

# Make the entrypoint script executable
# RUN chmod +x /app/entrypoint.sh

# Use entrypoint to set up model symlinks, then run the filter
# ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["python", "-m", "filter_huggingface_vision.filter"]
