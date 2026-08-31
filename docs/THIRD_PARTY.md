# Third-Party Code

LidarGait++ uses the official OpenGait implementation:

```text
repository: https://github.com/ShiqiYu/OpenGait.git
commit: f754f6f3831e9f83bb28f4e2f63dd43d8bcf9dc4
model SHA-256: 180cdb41da76cdd8950159469e72c525b4e555d1060edc89f3ee887f96aae748
utility SHA-256: 15ff5ff8f904f5b13d35d5d86f35d36814202769aed2bb0c12434acf43b881f8
```

Run `lidargaitpp/scripts/fetch_opengait.sh` to fetch and verify the pinned
checkout. The included compatibility patch only adds deterministic seed offset
support and tolerates an unavailable optional torchvision visualization import;
it does not modify the official LidarGait++ model files.

Review and preserve the upstream OpenGait license and citation requirements
before redistribution.
