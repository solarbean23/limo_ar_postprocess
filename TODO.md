# TODO

## LIMO AR Post-Processing Skill

추후 Claude에게 아래 목적의 Codex/Claude skill 생성을 요청할 예정.

- 입력: `lrf`, `mp4`, `db3`, `metadata.yaml`
- 처리:
  - ROS 2 db3 bag 및 metadata 파싱
  - 로봇 pose, fire state, target state, base timeline 구성
  - video-bag time offset 동기화
  - homography 기반 AR overlay 적용
  - `ar_only`, `wide_with_map` 후처리 렌더링
  - 짧은 preview 생성 후 최종 MP4/WebM 출력
- 출력: 입력 파일 세트를 기반으로 한 LIMO AR 후처리 영상

목표는 사용자가 실제 데이터 파일만 지정하면, 별도 UI나 ROS realtime node 없이 후처리 영상을 재현 가능하게 만드는 최소 skill이다.
