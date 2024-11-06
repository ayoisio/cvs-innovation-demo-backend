# Medical Content Review App (Backend)

The app uses several Google Cloud services, including:

* **Vertex AI:** For generative AI, including the Gemini model, and grounding with Google Search.
* **Cloud Tasks:** For asynchronous processing of chat requests.
* **Cloud Storage:** For storing uploaded media files.
* **Firestore:** For storing chat history, processed claims, and imprecise language instances.
* **Firebase Authentication:** For user authentication.
* **Cloud Run:** For hosting the backend service.

The backend code is written in Python and uses the Flask framework.

**Key functionalities and code components:**

1. **Authentication:** The `/chat` endpoint verifies user authentication using Firebase Authentication.

2. **Chat Request Handling:**
   * The `/chat` endpoint receives chat requests, extracts text and metadata, and creates a Cloud Task for background processing.
   * The `/chat/task` endpoint handles the actual chat processing. It uses Gemini with function calling and grounding with Google Search to:
     * Identify medical claims.
     * Identify imprecise language.
     * Generate alternative claims and improvement suggestions.
   * The results are stored in Firestore.

3. **Chat Title Generation:**
   * The `/chat/title` endpoint generates a title for the chat using Anthropic's generative AI.
   * The prompt for title generation is fetched from Google Cloud's Remote Config service.

4. **Asynchronous Processing:** Cloud Tasks are used to handle chat processing in the background, allowing for faster response times to the user.

5. **Data Storage:**
   * Firestore stores chat history, processed claims, and imprecise language instances.
   * Cloud Storage stores uploaded media files.

6. **Grounding:** The app uses grounding with Google Search to validate medical claims and provide citations.

7. **Function Calling:** Gemini's function calling feature is used to structure the chat processing workflow and call specific functions for medical claim identification and imprecise language identification.

8. **Containerization:** The backend is containerized using Docker and deployed on Cloud Run.


This backend architecture allows for efficient and scalable processing of medical content review requests, leveraging the power of generative AI and other Google Cloud services.
