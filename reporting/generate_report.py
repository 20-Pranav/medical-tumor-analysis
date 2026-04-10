from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

def create_tumor_report(comparison_result, confidence_scores, output_pdf='tumor_report.pdf'):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    y = height - 50
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Tumor Progression/Regression Report")
    y -= 30
    
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 25
    c.drawString(50, y, f"Old Volume: {comparison_result['old_volume_mm3']:.2f} mm³")
    y -= 20
    c.drawString(50, y, f"New Volume: {comparison_result['new_volume_mm3']:.2f} mm³")
    y -= 20
    c.drawString(50, y, f"Change: {comparison_result['change_mm3']:.2f} mm³ ({comparison_result['percent_change']:.1f}%)")
    y -= 20
    c.drawString(50, y, f"Status: {comparison_result['status']}")
    y -= 25
    c.drawString(50, y, f"Segmentation Confidence (mean mask IoU): {confidence_scores['mean']:.2f}")
    
    c.save()
    print(f"✅ Report saved to {output_pdf}")

if __name__ == "__main__":
    demo_result = {
        'old_volume_mm3': 14137.17,
        'new_volume_mm3': 4188.79,
        'change_mm3': -9948.38,
        'percent_change': -70.4,
        'status': 'Regression'
    }
    demo_conf = {'mean': 0.85}
    create_tumor_report(demo_result, demo_conf, 'demo_report.pdf')
