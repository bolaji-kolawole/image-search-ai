import express from 'express';
import type { Express, Request, Response } from 'express';
import multer from 'multer';
import sharp from 'sharp';
import * as tf from '@tensorflow/tfjs-node';
import { MongoClient, ObjectId } from 'mongodb';
import dotenv from 'dotenv';
import pino from 'pino';
import path from 'path';

dotenv.config();

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
});

const PORT = process.env.PORT || 3000;
const MONGODB_URL = process.env.MONGODB_URL || 'mongodb://localhost:27017';
const MONGODB_DB = process.env.MONGODB_DB || 'image_search';

const ALLOWED_EXTENSIONS = new Set([
  'jpg',
  'jpeg',
  'png',
  'webp',
]);

const MAX_FILE_SIZE = 10 * 1024 * 1024;

// ======================================================
// MULTER
// ======================================================

const upload = multer({
  storage: multer.memoryStorage(),

  fileFilter: (req: any, file: any, cb: any) => {
    const ext = path.extname(file.originalname)
      .toLowerCase()
      .slice(1);

    if (!ALLOWED_EXTENSIONS.has(ext)) {
      return cb(new Error('Invalid file type'));
    }

    cb(null, true);
  },

  limits: {
    fileSize: MAX_FILE_SIZE,
  },
});

// ======================================================
// TYPES
// ======================================================

interface ImageDocument {
  _id?: ObjectId;
  filename: string;
  vector: number[];
  createdAt?: Date;
}

interface SearchResult {
  index: number;
  similarity: number;
}

// ======================================================
// VECTOR SEARCH ENGINE
// ======================================================

class SimpleVectorIndex {
  private vectors: number[][] = [];

  ntotal = 0;

  add(vectors: number[][]) {
    this.vectors.push(...vectors);
    this.ntotal = this.vectors.length;
  }

  // Normalize vector
  private normalize(vector: number[]) {
    const magnitude = Math.sqrt(
      vector.reduce((sum, val) => sum + val * val, 0),
    );

    return vector.map((v) => v / magnitude);
  }

  // Cosine similarity
  private cosineSimilarity(a: number[], b: number[]) {
    const aNorm = this.normalize(a);
    const bNorm = this.normalize(b);

    let dot = 0;

    for (let i = 0; i < aNorm.length; i++) {
      dot += aNorm[i] * bNorm[i];
    }

    return dot;
  }

  search(queryVector: number[], topK: number): SearchResult[] {
    const similarities = this.vectors.map((vector, index) => {
      const similarity = this.cosineSimilarity(
        queryVector,
        vector,
      );

      return {
        index,
        similarity,
      };
    });

    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }
}

// ======================================================
// IMAGE SEARCH SERVICE
// ======================================================

class ImageSearchService {
  private vectorIndex!: SimpleVectorIndex;

  private imageIds: string[] = [];

  private db: MongoClient | null = null;

  private collection: any = null;

  isInitialized = false;

  async initialize() {
    this.vectorIndex = new SimpleVectorIndex();

    this.db = new MongoClient(MONGODB_URL);

    await this.db.connect();

    const database = this.db.db(MONGODB_DB);

    this.collection =
      database.collection<ImageDocument>('images');

    await this.collection.createIndex({ _id: 1 });

    this.isInitialized = true;

    logger.info('Service initialized');
  }

  // ======================================================
  // IMAGE ENCODING
  // ======================================================

  async encodeImage(buffer: Buffer): Promise<number[]> {
    // Resize image
    await sharp(buffer)
      .resize(224, 224)
      .removeAlpha()
      .raw()
      .toBuffer();

    /**
     * Replace this placeholder with real embedding model
     * Example:
     * MobileNet
     * CLIP
     * ResNet
     */

    const tensor = tf.randomNormal([512]);

    const vector = Array.from(tensor.dataSync());

    tensor.dispose();

    return vector;
  }

  // ======================================================
  // ADD IMAGE
  // ======================================================

  async addImage(
    filename: string,
    vector: number[],
  ): Promise<string> {
    const doc: ImageDocument = {
      filename,
      vector,
      createdAt: new Date(),
    };

    const result = await this.collection.insertOne(doc);

    this.vectorIndex.add([vector]);

    this.imageIds.push(result.insertedId.toString());

    return result.insertedId.toString();
  }

  // ======================================================
  // SEARCH
  // ======================================================

  async searchImages(
    vector: number[],
    topK: number,
  ) {
    if (this.vectorIndex.ntotal === 0) {
      return [];
    }

    const matches = this.vectorIndex.search(
      vector,
      topK,
    );

    const results = [];

    for (const match of matches) {
      const doc = await this.collection.findOne({
        _id: new ObjectId(
          this.imageIds[match.index],
        ),
      });

      if (!doc) continue;

      // Convert cosine similarity to percentage
      const percentage = Math.max(
        0,
        Math.min(
          100,
          Number((match.similarity * 100).toFixed(2)),
        ),
      );

      results.push({
        id: doc._id.toString(),
        filename: doc.filename,
        similarity: percentage,
        confidence:
          percentage >= 90
            ? 'Very High'
            : percentage >= 75
            ? 'High'
            : percentage >= 60
            ? 'Medium'
            : 'Low',
      });
    }

    return results;
  }

  // ======================================================
  // STATS
  // ======================================================

  async getStats() {
    const totalImages =
      await this.collection.countDocuments();

    return {
      totalImages,
      indexedImages: this.vectorIndex.ntotal,
    };
  }

  async shutdown() {
    if (this.db) {
      await this.db.close();
    }
  }
}

// ======================================================
// EXPRESS APP
// ======================================================

const app: Express = express();

const service = new ImageSearchService();

app.use(express.json());

// ======================================================
// HEALTH
// ======================================================

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    initialized: service.isInitialized,
  });
});

// ======================================================
// UPLOAD
// ======================================================

app.post(
  '/upload',
  upload.single('file'),
  async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res
          .status(400)
          .json({ error: 'No file uploaded' });
      }

      const vector = await service.encodeImage(
        req.file.buffer,
      );

      const id = await service.addImage(
        req.file.originalname,
        vector,
      );

      const stats = await service.getStats();

      return res.status(201).json({
        message: 'Image uploaded successfully',
        id,
        filename: req.file.originalname,
        totalImages: stats.totalImages,
      });
    } catch (error: any) {
      logger.error(error);

      return res.status(500).json({
        error: error.message,
      });
    }
  },
);

// ======================================================
// SEARCH
// ======================================================

app.post(
  '/search',
  upload.single('file'),
  async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res
          .status(400)
          .json({ error: 'No file uploaded' });
      }

      const vector = await service.encodeImage(
        req.file.buffer,
      );

      const topK = Math.min(
        parseInt(req.query.top_k as string) || 5,
        100,
      );

      const results = await service.searchImages(
        vector,
        topK,
      );

      return res.json({
        queryFilename: req.file.originalname,
        totalMatches: results.length,
        matches: results,
      });
    } catch (error: any) {
      logger.error(error);

      return res.status(500).json({
        error: error.message,
      });
    }
  },
);

// ======================================================
// STATS
// ======================================================

app.get('/stats', async (req, res) => {
  res.json(await service.getStats());
});

// ======================================================
// START SERVER
// ======================================================

async function start() {
  try {
    await service.initialize();

    app.listen(PORT, () => {
      logger.info(
        `Server running on port ${PORT}`,
      );
    });
  } catch (error) {
    logger.error(error);

    process.exit(1);
  }
}

start();