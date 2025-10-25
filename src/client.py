#!/usr/bin/env python3
"""
PDF to HTML Converter Client
Upload PDF files to the server and download HTML ZIP results
"""

import requests
import time
import os
import sys
from pathlib import Path

class PDFConverterClient:
    def __init__(self):
        # Hardcoded server URL
        self.server_url = "http://192.168.68.138:8000" 
        self.session = requests.Session()
    
    def check_server_health(self):
        """Check if server is running"""
        try:
            response = self.session.get(f"{self.server_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"Server is running: {data.get('message', 'Unknown')}")
                print(f"Version: {data.get('version', 'Unknown')}")
                return True
            else:
                print(f"Server responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to server at {self.server_url}")
            print("Make sure the server is running!")
            return False
        except Exception as e:
            print(f"Error checking server: {e}")
            return False
    
    def upload_pdf(self, pdf_path):
        """Upload PDF file and return job_id"""
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            return None
        
        if not pdf_path.lower().endswith('.pdf'):
            print(f"File must be a PDF: {pdf_path}")
            return None
        
        # Check file size
        file_size = os.path.getsize(pdf_path)
        if file_size > 50 * 1024 * 1024:
            print(f"File too large: {file_size / (1024*1024):.1f}MB (max 50MB)")
            return None
        
        print(f"Uploading {os.path.basename(pdf_path)} ({file_size / (1024*1024):.1f}MB)...")
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
                response = self.session.post(f"{self.server_url}/upload", files=files)
            
            if response.status_code == 200:
                data = response.json()
                job_id = data['job_id']
                print(f"Upload successful! Job ID: {job_id}")
                print(f"Status: {data['message']}")
                return job_id
            elif response.status_code == 429:
                print("Server is busy processing another PDF. Please try again later.")
                return None
            else:
                print(f"Upload failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Upload error: {e}")
            return None
    
    def get_status(self, job_id, retry_count=10):
        """Get job status with extended retry logic for processing periods"""
        for attempt in range(retry_count):
            try:
                response = self.session.get(f"{self.server_url}/status/{job_id}", timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    print(f"Job not found: {job_id}")
                    return None
                else:
                    print(f"Status check failed: {response.status_code}")
                    return None
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                wait_time = min(10 + (attempt * 2), 30)
                if attempt < retry_count - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    return {"status": "processing", "message": "Server processing continues"}
            except Exception as e:
                wait_time = min(5 + (attempt * 2), 20)
                if attempt < retry_count - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    return {"status": "processing", "message": "Processing continues"}
        return {"status": "processing", "message": "Monitoring continues"}
    
    def wait_for_completion(self, job_id, check_interval=10, max_wait_time=1800):
        """Wait for job to complete - processing can take a long time"""
        print(f"Waiting for processing to complete...")
        print(f"Checking status every {check_interval} seconds...")
        print(f"Will wait up to {max_wait_time//60} minutes")
        
        start_time = time.time()
        last_message = ""
        connection_issues_count = 0
        successful_checks = 0
        
        while True:
            elapsed = int(time.time() - start_time)
            
            # Check if we've exceeded max wait time
            if elapsed > max_wait_time:
                print(f"Maximum wait time ({max_wait_time//60} minutes) exceeded")
                print(f"Attempting to download result anyway...")
                return "timeout"
            
            status_data = self.get_status(job_id)
            
            # Handle case where status check completely fails
            if not status_data:
                connection_issues_count += 1
                print(f"[{elapsed:02d}s] Monitoring continues...")
                time.sleep(check_interval * 2)
                continue
            
            successful_checks += 1
            status = status_data['status']
            message = status_data.get('message', '')
            
            # Handle connection issues case but keep monitoring
            if "Connection" in message or "connection" in message:
                connection_issues_count += 1
                if connection_issues_count % 5 == 1:
                    print(f"[{elapsed:02d}s] Server processing continues...")
                time.sleep(check_interval + 5)
                continue
            
            # Only print status updates when they change or every 2 minutes
            time_since_last_update = elapsed % 120
            if message != last_message or time_since_last_update < check_interval:
                if message != last_message:
                    print(f"[{elapsed:02d}s] {message}")
                elif time_since_last_update < check_interval:
                    print(f"[{elapsed:02d}s] Still {status}...")
                last_message = message
            
            # Show progress if available
            if 'progress' in status_data:
                progress = status_data['progress']
                current = progress['current_page']
                total = progress['total_pages']
                percentage = progress['percentage']
                print(f"Progress: {current}/{total} pages ({percentage}%) - Processing page {current}")
            
            if status == 'completed':
                print(f"Processing completed in {elapsed} seconds ({elapsed//60}m {elapsed%60}s)!")
                return True
            elif status == 'failed':
                print(f"Server reported processing failed: {message}")
                print(f"Attempting download anyway...")
                return "failed_but_try_download"
            
            # Adaptive sleep
            if 'DotsOCR' in message or 'Processing' in message:
                sleep_time = check_interval + 5
            else:
                sleep_time = check_interval
            
            time.sleep(sleep_time)
    
    def download_result(self, job_id, output_dir=".", retry_count=15):
        """Download the result ZIP file with extended retry logic"""
        print(f"Starting download attempts...")
        
        for attempt in range(retry_count):
            wait_time = min(10 + (attempt * 3), 60)
            
            try:
                print(f"Download attempt {attempt + 1}/{retry_count}...")
                response = self.session.get(f"{self.server_url}/download/{job_id}", timeout=60)
                
                if response.status_code == 200:
                    # Get filename from response headers with robust parsing
                    content_disposition = response.headers.get('content-disposition', '')
                    filename = None
                    
                    # Debug: show what we received
                    print(f"Content-Disposition header: {content_disposition}")
                    
                    if content_disposition:
                        import re
                        from urllib.parse import unquote
                        
                        # Try RFC 5987 format first: filename*=charset'language'encoded-value
                        rfc5987_match = re.search(r"filename\*=([^']*)'([^']*)'(.+)", content_disposition)
                        if rfc5987_match:
                            charset = rfc5987_match.group(1)
                            language = rfc5987_match.group(2) 
                            encoded_filename = rfc5987_match.group(3)
                            # URL decode the filename
                            filename = unquote(encoded_filename)
                            print(f"RFC 5987 format detected - Charset: {charset}, Encoded: {encoded_filename}")
                            print(f"Decoded filename: {filename}")
                        else:
                            # Try regular format: filename="something" or filename=something
                            regular_match = re.search(r'filename=["\']?([^"\';\r\n]+)["\']?', content_disposition)
                            if regular_match:
                                filename = regular_match.group(1).strip()
                                print(f"Regular format - Extracted filename: {filename}")
                    
                    # Fallback to job ID if no filename found
                    if not filename:
                        filename = f"result_{job_id}.zip"
                        print(f"Using fallback filename: {filename}")
                    
                    print(f"Final download filename: {filename}")
                    
                    # Ensure output directory exists
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, filename)
                    
                    # Write file
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    
                    file_size = len(response.content)
                    print(f"Download successful: {output_path} ({file_size / (1024*1024):.1f}MB)")
                    return output_path
                
                elif response.status_code == 400:
                    print(f"Result not ready yet, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    print(f"Result file not found yet, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Server returned {response.status_code}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                print(f"Connection issue during download, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            except Exception as e:
                print(f"Download error: {e}")
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
        
        # After all retries failed
        print(f"Download failed after {retry_count} attempts")
        print(f"Manual download links:")
        print(f"Status: {self.server_url}/status/{job_id}")
        print(f"Download: {self.server_url}/download/{job_id}")
        return None
    
    def process_pdf(self, pdf_path, check_interval=10):
        """Complete workflow: upload, wait, download - output goes to same directory as PDF"""
        # Get directory of the PDF file for output
        pdf_directory = os.path.dirname(os.path.abspath(pdf_path))
        if not pdf_directory:
            pdf_directory = "."
        
        print(f"Starting PDF to HTML conversion...")
        print(f"Input: {pdf_path}")
        print(f"Output: {pdf_directory}")
        print(f"IMPORTANT: Processing can take several minutes for large PDFs")
        print("-" * 70)
        
        # Check server
        if not self.check_server_health():
            return False
        
        # Upload
        job_id = self.upload_pdf(pdf_path)
        if not job_id:
            return False
        
        print(f"Job ID: {job_id}")
        print(f"Manual monitoring URLs:")
        print(f"Status: {self.server_url}/status/{job_id}")
        print(f"Download: {self.server_url}/download/{job_id}")
        print("-" * 70)
        
        # Wait for completion
        completion_result = self.wait_for_completion(job_id, check_interval)
        
        print("-" * 70)
        
        # Handle different completion results
        if completion_result == True:
            print(f"Server reported completion - proceeding to download...")
        elif completion_result == "timeout":
            print(f"Monitoring timeout reached - attempting download anyway...")
        elif completion_result == "failed_but_try_download":
            print(f"Server reported failure - attempting download anyway...")
        else:
            print(f"Status monitoring had issues - attempting download anyway...")
        
        # Always try to download
        print(f"Starting download phase...")
        result_path = self.download_result(job_id, pdf_directory)
        
        if result_path:
            print(f"SUCCESS! HTML files saved to: {result_path}")
            print(f"You can now extract and view the HTML files")
            return True
        else:
            print(f"Could not download result after extensive attempts")
            print(f"Processing might still be running or completed on server")
            print(f"Try these URLs manually in a few minutes:")
            print(f"Status: {self.server_url}/status/{job_id}")
            print(f"Download: {self.server_url}/download/{job_id}")
            return False

def main():
    """Main function with command line interface"""
    print("PDF to HTML Converter Client")
    print("=" * 40)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python client.py <pdf_file>")
        print("Example: python client.py document.pdf")
        print("Note: Output will be saved in the same directory as the PDF")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Create client with hardcoded server URL
    client = PDFConverterClient()
    
    # Process PDF (output goes to same directory as PDF)
    success = client.process_pdf(pdf_path)
    
    if success:
        print("\nConversion completed successfully!")
        sys.exit(0)
    else:
        print("\nConversion failed!")
        sys.exit(1)

def interactive_mode():
    """Interactive mode for user-friendly experience"""
    print("PDF to HTML Converter Client - Interactive Mode")
    print("Server: http://103.227.97.76:8080")
    print("Output will be saved in same directory as PDF")
    print("=" * 50)
    
    # Create client with hardcoded server URL
    client = PDFConverterClient()
    
    # Check server
    if not client.check_server_health():
        input("Press Enter to exit...")
        return
    
    while True:
        print("\n" + "=" * 50)
        
        # Get PDF file
        pdf_path = input("Enter PDF file path (or 'quit' to exit): ").strip().strip('"\'')
        
        if pdf_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not pdf_path:
            continue
        
        # Process PDF (output automatically goes to PDF's directory)
        success = client.process_pdf(pdf_path)
        
        if success:
            print("\nConversion completed!")
        else:
            print("\nConversion failed!")
        
        # Ask if user wants to continue
        continue_choice = input("\nProcess another PDF? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("Goodbye!")
            break

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        interactive_mode()
    else:
        # Arguments provided - run command line mode
        main()