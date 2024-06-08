FROM neo4j:5.20.0-community

# Update the package list and install netcat and Python
RUN apt-get update && apt-get install -y \
    netcat \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Optionally upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory to /var/lib/neo4j (default for Neo4j in Docker)
WORKDIR /var/lib/neo4j

# Copy your Python scripts or applications into the container
COPY docker/services/neo4j ./repo

# Install any Python dependencies
RUN python3 -m pip install -r repo/requirements.txt

RUN chmod +x repo/scripts/entrypoint.sh

# Keep the original CMD or ENTRYPOINT of the base image
#CMD ["python3", "repo/scripts/ingest_neo4j.py", "--csv_dir", "repo/nba_v3/"]

ENTRYPOINT ["repo/scripts/entrypoint.sh"]
CMD ["neo4j"]
