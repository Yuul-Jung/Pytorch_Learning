"""05차시: GPU 없이 실행 가능한 저장 비트 수 계산 예제."""
PARAMETERS = 500_000_000  # 실제 모델 파라미터 수와 구별한 교육용 가정
for bits in (32, 16, 8, 4):
    byte_count = PARAMETERS * bits / 8
    print(f"{bits:>2}비트: {byte_count / 1e9:.3f} GB (십진 단위)")
print("실제 메모리에는 스케일·메타데이터·활성값·런타임 오버헤드가 추가됩니다.")