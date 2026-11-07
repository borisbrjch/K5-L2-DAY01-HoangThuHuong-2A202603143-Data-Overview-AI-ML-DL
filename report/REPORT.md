## Report
**Ngày chạy:**
6/11/2026

**Runtime Colab:** CPU/GPU
CPU

**Python / PyTorch / Ultralytics:**
Python: 3.13.18
PyTorch: 2.11.0+cpu
Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

## 1. Phân loại ảnh
1. Chép record hạng 1:
 {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  }

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

Record nói về cả bức ảnh, vì task là image_classification và không có thông tin bounding box hay vị trí của vật thể riêng lẻ.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

Danh sách class được quy định bởi taxonomy ImageNet-1K và model checkpoint yolo11n-cls.pt, không phải do model tự nghĩ ra.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

class_id (468) là mã định danh, class_name (cab) là tên lớp, còn taxonomy_name (ImageNet-1K) xác định hệ thống phân loại. Ba trường giúp xác định chính xác class và tránh nhầm lẫn giữa các taxonomy.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định tiêu chí chọn một lớp đại diện cho toàn bộ ảnh, chẳng hạn đối tượng chính, đối tượng chiếm ưu thế hoặc chủ đề trọng tâm của ảnh.

6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ là điểm số model đưa ra cho dự đoán, không đảm bảo dự đoán đúng. Ground truth phải là nhãn chuẩn được xác định qua quy trình annotation và review của con người.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
 {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  },
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

Bouding box xử dụng định dạng xyxy, gồm bốn toạ độ:
[x_min, y_min, x_max, y_max]

- Góc trên bên phải: toạ độ (385.33, 69.24)
- Góc dưới bên phải: toạ độ (498.92, 348.92)
- Chiều rộng: 113.58 px
- Chiều cao: 279.68 px


3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

- Confidence threshold 0.35 có 11 prediction
- Confidence threshold 0.60 có 6 prediction

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

Khi giảm confidence threshold, model sẽ giữ lại nhiều vật thể hơn, bao gồm cả những prediction có độ tin cậy thấp.
Điều này làm tăng khối lượng công việc của reviewer. 
Reviewer cần:
- Kiểm tra những bounding box mới được giữ lại để xác định chúng có thực sự chứa đối tượng hay không.
- Loại bỏ false positives (những prediction nhận diện nhầm vật thể hoặc nhận diện nhầm class).
- Kiểm tra các bounding box bị trùng lặp hoặc có vị trí không chính xác.
- Xác minh class label có đúng với đối tượng thực tế không.
- Điều chỉnh bounding box nếu nó không bao phủ đúng đối tượng.
Kết luận: Hạ threshold có thể giúp hạn chế bỏ sót đối tượng (false negatives), nhưng đổi lại reviewer phải kiểm tra nhiều prediction hơn để duy trì chất lượng dữ liệu.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Bounding box phải bao phủ toàn bộ phần đối tượng có thể quan sát được trong ảnh, sử dụng hình chữ nhật nhỏ nhất có thể nhưng vẫn chứa đầy đủ các phần nhìn thấy của đối tượng. Không tự ý mở rộng box sang những vùng nền xung quanh hoặc phần đối tượng không thể quan sát được.

Mục tiêu là đảm bảo tính nhất quán giữa các annotator, hạn chế bounding box quá rộng hoặc quá hẹp, đồng thời giúp dữ liệu huấn luyện có chất lượng tốt hơn.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Một đối tượng đáng chú ý là oven (lò nướng).
Trong JSON, prediction của đối tượng này có bounding box:
[0.11, 187.54, 195.96, 292.81]
Tọa độ x_min = 0.11 cho thấy bounding box gần như chạm vào mép trái của ảnh (x = 0).
Đây là một trường hợp đáng kiểm tra vì đối tượng có thể bị cắt một phần bởi ranh giới ảnh.

Escalation lên Lab Coach.

Lý do:
- Chỉ từ JSON chưa thể xác nhận chính xác phần nào của lò nướng bị cắt vì file không chứa pixel ảnh gốc.
- Nếu đối tượng thực sự bị cắt mép, cần xác nhận cách xử lý phần không nhìn thấy.
- Việc tự đưa ra một quy tắc chung khi chưa biết chính sách annotation có thể khiến các annotator xử lý không nhất quán.
- Lab Coach có thể xác nhận quy tắc chính thức trước khi áp dụng cho những trường hợp tương tự.


## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

{
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-001",
    "class_id": 0,
    "class_name": "person",
    "score": 0.899318,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.45,
      66.44,
      498.02,
      348.58
    ],
    "polygon_point_count": 348,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      ...
    ]
}
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

- Bounding Box (Box): Cho biết vật thể nằm ở đâu và chiếm diện tích bao nhiêu trong ảnh.

- Polygon (polygon_xy): Cho biết hình dạng và đường biên thực tế của vật thể.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

instance_id dùng để định danh một đối tượng cụ thể trong một tập dữ liệu hoặc một kết quả annotation.
Trong khi class ID dùng để phân loại vật thể theo category, ví dụ person: class ID 45, bowl: class ID 10,...
instance_id cũng phân loại vật thể nhưng phân theo từng cá thể, ví dụ hai chiếc bát nằm cạnh nhau có class ID giống nhau và đều là bowl, nhưng sẽ có instance_id khác nhau.
Ngoài ra còn có tracking ID dùng để trả lời câu hỏi "đây có phải cùng một đối tượng xuất hiện ở nhiều frame khác nhau không?" 

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

Người gán nhãn phải vẽ đường biên mask bám sát đường viền thực tế của từng vật thể, chỉ bao phủ những pixel thuộc về vật thể đó và không bao gồm background hoặc các vật thể khác.

Các nguyên tắc thực hiện:
- Bám sát đường viền: Vẽ theo hình dạng thực tế của vật thể, không dùng hình chữ nhật thay thế.
- Không bao gồm background: Không đưa vùng nền hoặc khoảng trống bên trong vật thể vào mask.
- Vật thể bị che khuất: Chỉ vẽ phần vật thể thực sự nhìn thấy; không tự suy đoán và vẽ phần bị che khuất.
- Các vật thể cùng class: Phải tạo mask riêng cho từng instance, kể cả khi chúng tiếp xúc hoặc chồng lấn nhau.
- Độ chính xác: Tránh vẽ quá rộng, quá hẹp hoặc tạo các điểm biên không cần thiết.

Tiêu chí nghiệm thu: Mask phải đại diện chính xác cho vùng vật thể quan sát được, không lẫn pixel của vật thể khác hoặc background.

Ví dụ: Hai chiếc bát nằm cạnh nhau phải có hai mask riêng biệt, dù chúng cùng thuộc class bowl.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Ở đây tôi chọn ra một vật thể bowl, có hình bị crop và khuất mất một góc.
{
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-006",
    "class_id": 45,
    "class_name": "bowl",
    "score": 0.528202,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      0.11,
      302.85,
      85.7,
      370.15
    ],
    "polygon_point_count": 84,
    "polygon_xy": 
    [
        29.0,
        303.0
      ],
      [
        28.0,
        304.0
      ],
      [
        26.0,
        304.0
      ],
      [
        25.0,
        305.0
      ],
      [
        23.0,
        305.0
      ],
      [
        22.0,
        306.0
      ],
      [
        20.0,
        306.0
      ],
      [
        19.0,
        307.0
      ],
    }

Khi gặp những trường hợp bất thường, tôi sẽ đối chiếu lại giữa JSON và ảnh. Nếu guideline đã có hướng dẫn cho trường hợp này, thì follow. Còn nếu guideline không có trường hợp này thì tôi sẽ chọn escalate. Vừa để đảm bảo tiến độ công việc, nhưng cũng không làm ảnh hưởng ground truth.

Trong ví dụ với vật thể 'bowl' phía trên, ở đây không tìm thấy lỗi đặc biệt nghiêm trọng, và đây là một vật thể bị crop, có thể áp dụng guideline nếu cover trường hợp này. 

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn class của toàn bộ ảnh, ví dụ: cab, minibus, police_van. | Các phương tiện có đặc điểm tương tự dễ bị nhầm lẫn. Model có nhiều dự đoán với score khác nhau; JSON chỉ chứa prediction, không phải ground truth. | Quan sát ảnh, xác định class phù hợp theo taxonomy và guideline. Không lấy prediction của model làm ground truth. | Kiểm tra nhãn có đúng nội dung ảnh, đúng taxonomy và nhất quán với guideline hay không. |
| Phát hiện vật thể | Mỗi vật thể có một nhãn lớp và một bounding box; đơn vị nhãn là 'instance_id'. | Trong JSON kitchen, có các dự đoán ở nhiều mức confidence. Hạ threshold từ 0.35 xuống 0.20 làm xuất hiện thêm 6 dự đoán (spoon ×3, potted plant ×1, dining table ×1, bottle ×1). Một box oven có x_min = 0.11, sát mép trái ảnh; chỉ từ JSON chưa thể kết luận vật thể bị cắt hay box sai. | Vẽ một bounding box cho từng vật thể theo quy ước, gán đúng class; xử lý vật thể bị che khuất hoặc nằm sát mép ảnh theo guideline. | Kiểm tra class, số lượng instance và độ khớp của từng box; chú ý box bị thiếu/thừa, box lệch, vật thể nhỏ và trường hợp bị che khuất/cắt mép. |
| Instance segmentation | Mỗi vật thể có một class label và một mask/polygon riêng biệt. Mỗi instance được phân biệt bằng instance_id. | Instance kitchen-006 được dự đoán là bowl, có score = 0.528202. Bounding box bắt đầu tại x = 0.11, polygon có các điểm tại x = 1 như [1,310], [1,367], cho thấy mask nằm rất sát biên trái ảnh. Cần kiểm tra khả năng vật thể bị crop hoặc mask lấn vào background. | Đối chiếu ảnh gốc với polygon; vẽ mask theo đường biên thực tế của phần vật thể nhìn thấy. Nếu biên không thể xác định hoặc guideline không quy định cách xử lý, escalation cho reviewer. | Kiểm tra class, instance separation, polygon boundary, vùng mask có bao gồm background không và mask có nhất quán với guideline không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Không chia sẻ, sao chép hoặc sử dụng dữ liệu ngoài phạm vi cho phép của bài lab; chỉ truy cập dữ liệu phục vụ mục đích thực hành.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng xử lý và báo ngay cho người hướng dẫn hoặc người phụ trách chương trình để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
