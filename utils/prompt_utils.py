json_parsers_template = """
    From the weekly report below, extract (as many as possible) relationships between environmental events in Thai
    # if the event is not associated with numerical date, then assign blank
    # make sure to answer in the correct format. 'date' should be in the format YYYY-MM-DD in AD.

    Weekly report:
    {report}
    
    You should parse this in the following JSON Format in Thai: 
    {format_instructions}
    """
    
instruction_prompt = """
        กรุณาตอบคำถามเป็นภาษาไทยในลักษณะรายงานสรุป โดยพิจารณาตามบริบทที่ได้รับจาก Cypher Chain ดังนี้:

        1. หากไม่แน่ใจในคำตอบ
        - สามารถค้นหาเครื่องมือที่เกี่ยวข้องอีกครั้งเพื่อยืนยันคำตอบ
        - อย่าพยายามตอบจากมุมมองส่วนตัวของคุณ
        
        2. การขยายคำตอบ
        - พยายามอธิบายคำตอบโดยละเอียด และขยายความจากข้อมูลหรือบริบทที่ให้มา
        
        3. การตั้งคำถามย่อย
        - แบ่งคำถามออกเป็นคำถามย่อย เพื่อให้ง่ายต่อการค้นหาข้อมูล
        - ใช้คำเชื่อม OR หรือ CONTAINS เมื่อค้นหาข้อมูลที่มีหลายความหมายหรือคำใกล้เคียงเท่านั้น
        
        4. การจัดรูปแบบคำตอบ
        - แบ่งคำตอบออกเป็นส่วนต่าง ๆ เพื่อความสะดวกของผู้อ่าน
        
        ห้ามให้เกิด Thought ก่อน Action เด็ดขาด
    """