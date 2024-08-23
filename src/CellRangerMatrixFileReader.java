package src;
/*******************************************************************************
 * NGSEP - Next Generation Sequencing Experience Platform
 * Copyright 2016 Jorge Duitama
 *
 * This file is part of NGSEP.
 *
 *     NGSEP is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     NGSEP is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with NGSEP.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/


import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.logging.Logger;


/**
 * 
 * @author Jorge Duitama
 *
 */
public class CellRangerMatrixFileReader implements Iterable<CellRangerCount>,Closeable  {
	
	private Logger log = Logger.getAnonymousLogger();
	
	private BufferedReader in;
	
	private CellRangerMatrixIterator currentIterator = null;
	
	private int numRows = 0;
	
	private int numCols = 0;
	
	private int numEntries = 0;
	
	
	public CellRangerMatrixFileReader (String filename) throws IOException {
		init(null,new File(filename));
	}
	public CellRangerMatrixFileReader (File file) throws IOException {
		init(null,file);
	}
	public CellRangerMatrixFileReader (InputStream stream) throws IOException {
		init(stream,null);
	}
	
	public Logger getLog() {
		return log;
	}
	public void setLog(Logger log) {
		if (log == null) throw new NullPointerException("Log can not be null");
		this.log = log;
	}
	@Override
	public void close() throws IOException {
		in.close();
	}

	@Override
	public Iterator<CellRangerCount> iterator() {
		if (in == null) {
            throw new IllegalStateException("File reader is closed");
        }
        if (currentIterator != null) {
            throw new IllegalStateException("Iteration in progress");
        }
        currentIterator = new CellRangerMatrixIterator();
		return currentIterator;
	}
	
	private void init (InputStream stream, File file) throws IOException {
		if (stream != null && file != null) throw new IllegalArgumentException("Stream and file are mutually exclusive");
		if(file!=null) {
			stream = new FileInputStream(file);
			if(file.getName().toLowerCase().endsWith(".gz")) {
				stream = new ConcatGZIPInputStream(stream);
			}
		}
		in = new BufferedReader(new InputStreamReader(stream));
		
	}
	
	private CellRangerCount load (BufferedReader in) throws IOException {
		String line = in.readLine();
		//process header
		while(line!=null && line.charAt(0)=='%') {
			line = in.readLine();
		}
		if(line==null) return null;
		String [] vals = line.split(" ");
		if(numRows==0) {
			numRows = Integer.parseInt(vals[0]);
			numCols =  Integer.parseInt(vals[1]);
			numEntries = Integer.parseInt(vals[2]);
		}
		line = in.readLine();
		if(line==null) return null;
		vals = line.split(" ");
		return new CellRangerCount(Integer.parseInt(vals[1])-1, Integer.parseInt(vals[0])-1, Integer.parseInt(vals[2]));
	}
	
	
	private class CellRangerMatrixIterator implements Iterator<CellRangerCount> {
		private CellRangerCount nextRecord;
		public CellRangerMatrixIterator() {
			nextRecord = loadRecord();
		}
		@Override
		public boolean hasNext() {
			return nextRecord!=null;
		}

		@Override
		public CellRangerCount next() {
			if(nextRecord==null) throw new NoSuchElementException();
			CellRangerCount answer = nextRecord;
			nextRecord = loadRecord();
			return answer;
		}

		private CellRangerCount loadRecord() {
			CellRangerCount count;
			while(true) {
				try {
					count = load(in);
				} catch (IOException e) {
					throw new RuntimeException(e);
				}
				return count;
			} 
		}
		@Override
		public void remove() {
			throw new UnsupportedOperationException("Remove not supported by FastqFileIterator");
		}
	}
}
